from dataclasses import dataclass
from time import time
from typing import Tuple

from openvino.inference_engine import IECore
from openvino.inference_engine.ie_api import ExecutableNetwork


@dataclass
class OpenVinoModel:
    structure: str  # structure path
    weights: str  # model

    @classmethod
    def from_path(cls, path_prefix: str) -> "OpenVinoModel":
        return OpenVinoModel(
            structure=f"{path_prefix}.xml", weights=f"{path_prefix}.bin"
        )


def load_with_IECore(
    model_definition: OpenVinoModel,
    ie=IECore(),
    device_name: str = "CPU",
    num_requests: int = 1,
) -> Tuple[ExecutableNetwork, str]:
    """
    load a model using  IECore
    :param device_name: the device name ex: "CPU"
    :param model_definition: model definition as an OpenVinoModel
    :param num_request: number of the requests (default = 1)
    :return: the tuple (network, input_name)
    """
    net = ie.read_network(
        model=model_definition.structure, weights=model_definition.weights
    )
    exec_net = ie.load_network(
        network=net, device_name=device_name, num_requests=num_requests
    )
    input_layer = next(
        iter(net.input_info)
    )  # to check if it is equivalent to 2021 next(iter(model.inputs))
    # print(net.input_info[input_layer].input_data.shape)
    return exec_net, input_layer
