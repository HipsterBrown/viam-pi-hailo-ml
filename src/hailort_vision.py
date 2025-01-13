from functools import lru_cache
from typing import ClassVar, List, Literal, Mapping, Optional, Sequence, cast

from cv2.typing import MatLike
from typing_extensions import Self
from viam.components.camera import Camera
from viam.logging import getLogger
from viam.media.video import ViamImage
from viam.media.utils.pil import viam_to_pil_image
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.proto.service.vision import Classification, Detection
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.vision import Vision, CaptureAllResult
from viam.utils import ValueTypes, struct_to_dict

from picamera2.devices import Hailo

from pydantic import BaseModel, Field

import cv2
import numpy as np

LOGGER = getLogger(__name__)


class VisionConfig(BaseModel):
    model: Literal[
        "resnet_v1_50_h8l",
        "scrfd_2.5g_h8l",
        "yolov6n_h8",
        "yolov6n_h8l",
        "yolov8s_h8",
        "yolov8s_h8l",
        "yolox_s_leaky_h8l_rpi",
    ] = Field(default="yolov8s_h8l")
    labels_path: str = Field(default="assets/coco.txt")
    default_minimum_confidence: float = Field(le=1.0, ge=0.0, default=0.50)


class HailoRT(Vision, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily("hipsterbrown", "vision"), "hailo-rt")

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        """This method creates a new instance of this Vision service.
        The default implementation sets the name from the `config` parameter and then calls `reconfigure`.

        Args:
            config (ComponentConfig): The configuration for this resource
            dependencies (Mapping[ResourceName, ResourceBase]): The dependencies (both implicit and explicit)

        Returns:
            Self: The resource
        """
        return super().new(config, dependencies)

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Sequence[str]:
        """This method allows you to validate the configuration object received from the machine,
        as well as to return any implicit dependencies based on that `config`.

        Args:
            config (ComponentConfig): The configuration for this resource

        Returns:
            Sequence[str]: A list of implicit dependencies
        """
        VisionConfig(**struct_to_dict(config.attributes))
        return []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        """This method allows you to dynamically update your service when it receives a new `config` object.

        Args:
            config (ComponentConfig): The new configuration
            dependencies (Mapping[ResourceName, ResourceBase]): Any dependencies (both implicit and explicit)
        """
        self.DEPS = dependencies
        self.config = VisionConfig(**struct_to_dict(config.attributes))

        with open(self.config.labels_path, "r") as f:
            self.labels = f.read().splitlines()

        self.model_path = f"/usr/share/hailo-models/{self.config.model}.hef"
        self.threshold = self.config.default_minimum_confidence
        self.hailo = Hailo(self.model_path)

    async def get_camera_image(self, camera_name: str) -> ViamImage:
        camera_resource = self.DEPS.get(Camera.get_resource_name(camera_name))
        camera = cast(Camera, camera_resource)
        return await camera.get_image(mime_type="image/jpeg")

    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> CaptureAllResult:
        LOGGER.debug(
            f"Processing capture request for {camera_name} with arguments: return_image ({return_image}), return_classifications ({return_classifications}), return_detections ({ return_detections})"
        )
        properties = await self.get_properties()
        result = CaptureAllResult()
        image = await self.get_camera_image(camera_name)
        if return_image:
            result.image = await self.get_camera_image(camera_name)

        if return_detections and properties.detections_supported:
            result.detections = await self.get_detections(image)

        if return_classifications and properties.classifications_supported:
            result.classifications = await self.get_classifications(image, 1)

        return result

    def _get_label_for_index(self, index: int) -> str:
        labels = self._get_labels(self.config.labels_path)
        return labels[index]

    @lru_cache
    def _get_labels(self, _labels_path: str) -> List[str]:
        """labels_path is used to break the cache if the configuration changes"""
        labels = self.labels
        return labels

    def _extract_detections(
        self, hailo_output, width, height, resized_width, resized_height
    ) -> List[Detection]:
        results: List[Detection] = []
        scale = min(resized_width / width, resized_height / height)
        pad_width = (resized_width - int(width * scale)) // 2

        for class_id, detections in enumerate(hailo_output):
            for detection in detections:
                if len(detection) == 0:
                    continue

                score = detection[4]
                if score >= self.threshold:
                    y0, x0, y1, x1 = detection[:4]

                    x0 = int(x0 * resized_width)
                    y0 = int(y0 * resized_height)
                    x1 = int(x1 * resized_width)
                    y1 = int(y1 * resized_height)

                    x0 = max(0, int((x0 - pad_width) / scale))
                    y0 = max(0, int(y0 / scale))
                    x1 = min(width, int((x1 - pad_width) / scale))
                    y1 = min(height, int(y1 / scale))

                    results.append(
                        Detection(
                            x_min=x0,
                            y_min=y0,
                            x_max=x1,
                            y_max=y1,
                            confidence=score,
                            class_name=self._get_label_for_index(class_id),
                        )
                    )

        return results

    def _preprocess_image(self, image: ViamImage) -> tuple[MatLike, int, int, int, int]:
        input_height, input_width, _ = self.hailo.get_input_shape()
        image_height, image_width = image.height, image.width
        scale = min(input_width / image_width, input_height / image_height)
        new_width = int(image_width * scale)
        new_height = int(image_height * scale)

        LOGGER.debug(
            f"Input size: ({input_width, input_height}), Image size: ({image_height, image_width})"
        )
        pil_image = viam_to_pil_image(image)
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        del pil_image
        resized_image = cv2.resize(
            opencv_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC
        )
        padded_image = np.full(
            (input_height, input_width, 3), (0, 0, 0), dtype=np.uint8
        )
        x_offset = (input_width - new_width) // 2
        y_offset = (input_height - input_height) // 2
        padded_image[
            y_offset : y_offset + new_height, x_offset : x_offset + new_width
        ] = resized_image
        del opencv_image
        del resized_image

        return padded_image, image_width, image_height, input_width, input_height

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        return await self.get_detections(await self.get_camera_image(camera_name))

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        processed_image, og_width, og_height, input_width, input_height = (
            self._preprocess_image(image)
        )
        results = self.hailo.run(processed_image)
        del processed_image
        if len(results) == 1:
            results = results[0]

        detections = self._extract_detections(
            results, og_width, og_height, input_width, input_height
        )
        return detections

    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        raise NotImplementedError()

    async def get_classifications(
        self,
        image: ViamImage,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        raise NotImplementedError()

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[PointCloudObject]:
        raise NotImplementedError()

    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> Vision.Properties:
        properties = Vision.Properties()
        properties.object_point_clouds_supported = False
        properties.detections_supported = True
        properties.classifications_supported = False
        return properties

    async def close(self):
        self.hailo.close()
