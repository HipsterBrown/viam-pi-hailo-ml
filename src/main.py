import asyncio
from typing import ClassVar, Dict, Mapping, Optional, Sequence

from numpy.typing import NDArray
import numpy as np
from typing_extensions import Self
from viam.logging import getLogger
from viam.module.module import Module
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.proto.service.mlmodel import Metadata, TensorInfo
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.mlmodel import MLModel
from viam.utils import ValueTypes

from picamera2.devices import Hailo


LOGGER = getLogger("HailoRT")


class HailoRt(MLModel, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily("hipsterbrown", "mlmodel"), "hailo-rt")

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        """This method creates a new instance of this MLModel service.
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
        return []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        """This method allows you to dynamically update your service when it receives a new `config` object.

        Args:
            config (ComponentConfig): The new configuration
            dependencies (Mapping[ResourceName, ResourceBase]): Any dependencies (both implicit and explicit)
        """
        self.model_path = "/usr/share/hailo-models/yolov8s_h8l.hef"
        self.threshold = 0.5
        self.hailo = Hailo(self.model_path)

    async def infer(
        self,
        input_tensors: Dict[str, NDArray],
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, NDArray]:
        # return await self.hailo.run_async(input_tensors)
        LOGGER.debug(f"Input tensor keys: {list(input_tensors.keys())}")
        return {
            "location": np.ndarray([]),
            "category": np.ndarray([]),
            "score": np.ndarray([]),
        }

    async def metadata(
        self,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> Metadata:
        LOGGER.debug("Received metadata request")
        inputs, outputs = self.hailo.describe()
        LOGGER.debug("Result from Hailo describe method")
        LOGGER.debug(inputs)
        LOGGER.debug(outputs)
        input_info = [
            TensorInfo(
                name=name,
                data_type=str(data_type.name).lower(),
                shape=[1] + self._prepShape(shape),
            )
            for name, shape, data_type in inputs
        ]
        output_info = [
            TensorInfo(
                name=name,
                data_type=str(data_type.name).lower(),
                shape=self._prepShape(shape),
            )
            for name, shape, data_type in outputs
        ]
        return Metadata(
            name="yolo8s_h8l",
            description="YOLO 8S compiled for the Hailo 8L NPU",
            type="object_detector",
            input_info=input_info,
            output_info=output_info,
        )

    def _prepShape(self, tensorShape: tuple) -> list[int]:
        return [-1 if t is None else t for t in list(tensorShape)]

    async def close(self):
        return self.hailo.close()


if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
