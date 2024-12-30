import asyncio
from typing import ClassVar, Dict, Final, Mapping, Optional, Sequence

from numpy.typing import NDArray
from typing_extensions import Self
from viam.module.module import Module
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.proto.service.mlmodel import Metadata
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.mlmodel import *
from viam.utils import ValueTypes


class HailoRt(MLModel, EasyResource):
    MODEL: ClassVar[Model] = Model(
        ModelFamily("hipsterbrown", "pi-hailo-ml"), "hailo-rt"
    )

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
        return super().reconfigure(config, dependencies)

    async def infer(
        self,
        input_tensors: Dict[str, NDArray],
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, NDArray]:
        raise NotImplementedError()

    async def metadata(
        self,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> Metadata:
        raise NotImplementedError()


if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())

