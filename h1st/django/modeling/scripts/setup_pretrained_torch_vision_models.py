from pathlib import Path

from torchvision.models.alexnet import (alexnet,
                                        model_urls as alexnet_urls)
from torchvision.models.densenet import (densenet121, densenet161,
                                         densenet169, densenet201,
                                         model_urls as densenet_urls)
from torchvision.models.googlenet import (googlenet,
                                          model_urls as googlenet_urls)
from torchvision.models.inception import (inception_v3,
                                          model_urls as inception_urls)
from torchvision.models.mnasnet import (mnasnet0_5,   # mnasnet0_75,
                                        mnasnet1_0,   # mnasnet1_3,
                                        _MODEL_URLS as MNASNET_URLS)
from torchvision.models.mobilenetv2 import (mobilenet_v2,
                                            model_urls as mobilenet_v2_urls)
from torchvision.models.mobilenetv3 import (mobilenet_v3_large,
                                            mobilenet_v3_small,
                                            model_urls as mobilenet_v3_urls)
from torchvision.models.resnet import (resnet18, resnet34, resnet50,
                                       resnet101, resnet152,
                                       resnext50_32x4d, resnext101_32x8d,
                                       wide_resnet50_2, wide_resnet101_2,
                                       model_urls as resnet_urls)
from torchvision.models.shufflenetv2 import (shufflenet_v2_x0_5,
                                             shufflenet_v2_x1_0,
                                             # shufflenet_v2_x1_5,
                                             # shufflenet_v2_x2_0,
                                             model_urls as shufflenet_v2_urls)
from torchvision.models.squeezenet import (squeezenet1_0, squeezenet1_1,
                                           model_urls as squeezenet_urls)
from torchvision.models.vgg import (vgg11, vgg11_bn, vgg13, vgg13_bn,
                                    vgg16, vgg16_bn, vgg19, vgg19_bn,
                                    model_urls as vgg_urls)

from tqdm import tqdm

from ...util import fullqualname
from ..models import PreTrainedTorchVisionImageNetClassifier


MODEL_SPECS = [
    (alexnet, alexnet_urls[alexnet.__name__]),

    (densenet121, densenet_urls[densenet121.__name__]),
    (densenet161, densenet_urls[densenet161.__name__]),
    (densenet169, densenet_urls[densenet169.__name__]),
    (densenet201, densenet_urls[densenet201.__name__]),

    (googlenet, googlenet_urls[googlenet.__name__]),

    (inception_v3, inception_urls['inception_v3_google']),

    (mnasnet0_5, MNASNET_URLS[mnasnet0_5.__name__]),
    # (mnasnet0_75, MNASNET_URLS[mnasnet0_75.__name__]),
    (mnasnet1_0, MNASNET_URLS[mnasnet1_0.__name__]),
    # (mnasnet1_3, MNASNET_URLS[mnasnet1_3.__name__]),

    (mobilenet_v2, mobilenet_v2_urls[mobilenet_v2.__name__]),
    (mobilenet_v3_large, mobilenet_v3_urls[mobilenet_v3_large.__name__]),
    (mobilenet_v3_small, mobilenet_v3_urls[mobilenet_v3_small.__name__]),

    (resnet18, resnet_urls[resnet18.__name__]),
    (resnet34, resnet_urls[resnet34.__name__]),
    (resnet50, resnet_urls[resnet50.__name__]),
    (resnet101, resnet_urls[resnet101.__name__]),
    (resnet152, resnet_urls[resnet152.__name__]),
    (resnext50_32x4d, resnet_urls[resnext50_32x4d.__name__]),
    (resnext101_32x8d, resnet_urls[resnext101_32x8d.__name__]),
    (wide_resnet50_2, resnet_urls[wide_resnet50_2.__name__]),
    (wide_resnet101_2, resnet_urls[wide_resnet101_2.__name__]),

    (squeezenet1_0, squeezenet_urls[squeezenet1_0.__name__]),
    (squeezenet1_1, squeezenet_urls[squeezenet1_1.__name__]),

    (shufflenet_v2_x0_5, shufflenet_v2_urls['shufflenetv2_x0.5']),
    (shufflenet_v2_x1_0, shufflenet_v2_urls['shufflenetv2_x1.0']),
    # (shufflenet_v2_x1_5, shufflenet_v2_urls['shufflenetv2_x1.5']),
    # (shufflenet_v2_x2_0, shufflenet_v2_urls['shufflenetv2_x2.0']),

    (vgg11, vgg_urls[vgg11.__name__]),
    (vgg11_bn, vgg_urls[vgg11_bn.__name__]),
    (vgg13, vgg_urls[vgg13.__name__]),
    (vgg13_bn, vgg_urls[vgg13_bn.__name__]),
    (vgg16, vgg_urls[vgg16.__name__]),
    (vgg16_bn, vgg_urls[vgg16_bn.__name__]),
    (vgg19, vgg_urls[vgg19.__name__]),
    (vgg19_bn, vgg_urls[vgg19_bn.__name__]),
]


def run():
    model_name_prefix = f'{PreTrainedTorchVisionImageNetClassifier.__name__}-'

    for torch_model_loader, global_url in tqdm(MODEL_SPECS):
        print(PreTrainedTorchVisionImageNetClassifier.objects.update_or_create(
            name=model_name_prefix + torch_model_loader.__name__,
            defaults=dict(
                py_loader_module_and_qualname=fullqualname(torch_model_loader),
                artifact_global_url=global_url,
                artifact_local_path=(Path('~/.cache/torch/hub/checkpoints') /
                                     Path(global_url).name),
                params=dict(__init__=dict(pretrained=True,
                                          progress=True))))[0])
