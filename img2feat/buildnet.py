import torch
import torch.nn as nn
import torchvision

class BuildNet:
    @classmethod
    def __prefix(cls):
        return 'build_'

    @classmethod
    def available_names(cls):
        prefix = cls.__prefix()
        names = []
        for name in dir(cls):
            if( name.startswith(prefix) ):
                names.append( name[len(prefix):] )
        return names

    @classmethod
    def build( cls, name ):
        if( not ( name in cls.available_names() ) ):
            raise NotImplementedError( '**' + name + '** is not implemented.')
        print('eval : ', cls.__name__+'.'+cls.__prefix()+name)
        return eval(cls.__name__+'.'+cls.__prefix()+name)()

    '''
    build_[network name]():

    The output should be [N, ???, 1, 1] or [N, ???].
    ??? is the dimension of the feature

    The final layer may be
    nn.AdaptiveAvgPool2d(output_size=(1, 1)).

    '''

    @classmethod
    def build_alexnet( cls ):
        # model = torchvision.models.alexnet(pretrained=True) # Deprecated
        model = torchvision.models.alexnet(weights="IMAGENET1K_V1", progress=False)
        print('model loading done')
        modules=list(model.children())[:-2]
        modules.append( nn.AdaptiveAvgPool2d(output_size=(1, 1)) )
        network = nn.Sequential(*modules)
        return network, 256


    @classmethod
    def __vgg_p2p( cls, sequential ):
        k=1
        for module in sequential:
            if( isinstance( module, nn.modules.pooling.MaxPool2d) ):
                module.kernel_size=3
                module.stride=1
                module.padding=1
                k+=1
            elif( isinstance( module, nn.modules.conv.Conv2d ) ):
                module.dilation = (k,k)
                module.padding = (k,k)
        return sequential

    @classmethod
    def __vgg( cls, name, dim_feature, P2P ):
        model = eval('torchvision.models.'+name)(weights="IMAGENET1K_V1", progress=False)
        modules = list(model.children())[:-2]
        if( P2P ):
            cls.__vgg_p2p(modules[0])
        else:
            modules.append( nn.AdaptiveAvgPool2d(output_size=(1, 1)) )
        network = nn.Sequential(*modules)
        return network, dim_feature

    @classmethod
    def build_vgg11( cls ):
        return cls.__vgg( 'vgg11', 512, False )

    @classmethod
    def build_vgg13( cls ):
        return cls.__vgg( 'vgg13', 512, False )

    @classmethod
    def build_vgg16( cls ):
        return cls.__vgg( 'vgg16', 512, False )

    @classmethod
    def build_vgg19( cls ):
        return cls.__vgg( 'vgg19', 512, False )

    @classmethod
    def build_vgg11p2p( cls ):
        return cls.__vgg( 'vgg11', 512, True )

    @classmethod
    def build_vgg13p2p( cls ):
        return cls.__vgg( 'vgg13', 512, True )

    @classmethod
    def build_vgg16p2p( cls ):
        return cls.__vgg( 'vgg16', 512, True )

    @classmethod
    def build_vgg19p2p( cls ):
        return cls.__vgg( 'vgg19', 512, True )

    @classmethod
    def __resnet( cls, name, dim_feature ):
        model = eval('torchvision.models.'+name)(weights="IMAGENET1K_V1", progress=False)
        modules = list(model.children())[:-2]
        modules.append( nn.AdaptiveAvgPool2d(output_size=(1, 1)) )
        network = nn.Sequential(*modules)
        return network, dim_feature


    @classmethod
    def build_resnet18( cls ):
        return cls.__resnet( 'resnet18', 512 )

    @classmethod
    def build_resnet34( cls ):
        return cls.__resnet( 'resnet34', 512 )

    @classmethod
    def build_resnet101( cls ):
        return cls.__resnet( 'resnet101', 2048 )

    @classmethod
    def build_resnet152( cls ):
        return cls.__resnet( 'resnet152', 2048 )

    @classmethod
    def build_densenet121( cls ):
        model = torchvision.models.densenet121(weights="IMAGENET1K_V1", progress=False)
        modules=list(model.children())[:-1]
        print('-----------------------------------------------------------', GAP)
        if( GAP ):
            modules.append( nn.AdaptiveAvgPool2d(output_size=(1, 1)) )
        network = nn.Sequential(*modules)
        return network, 1024

    @classmethod
    def build_densenet169( cls ):
        model = torchvision.models.densenet169(weights="IMAGENET1K_V1", progress=False)
        modules=list(model.children())[:-1]
        modules.append( nn.AdaptiveAvgPool2d(output_size=(1, 1)) )
        network = nn.Sequential(*modules)
        return network, 1664

    @classmethod
    def build_densenet201( cls ):
        model = torchvision.models.densenet201(weights="IMAGENET1K_V1", progress=False)
        modules=list(model.children())[:-1]
        modules.append( nn.AdaptiveAvgPool2d(output_size=(1, 1)) )
        network = nn.Sequential(*modules)
        return network, 1920

    @classmethod
    def build_densenet161( cls ):
        model = torchvision.models.densenet161(weights="IMAGENET1K_V1", progress=False)
        modules=list(model.children())[:-1]
        modules.append( nn.AdaptiveAvgPool2d(output_size=(1, 1)) )
        network = nn.Sequential(*modules)
        return network, 2208

    @classmethod
    def build_googlenet( cls ):
        model = torchvision.models.googlenet(weights="IMAGENET1K_V1", progress=False)
        modules=list(model.children())[:-3]
        modules.append( nn.AdaptiveAvgPool2d(output_size=(1, 1)) )
        network = nn.Sequential(*modules)
        return network, 1024

    @classmethod
    def build_mobilenet( cls ):
        model = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1", progress=False)
        modules=list(model.children())[:-1]
        modules.append( nn.AdaptiveAvgPool2d(output_size=(1, 1)) )
        network = nn.Sequential(*modules)
        return network, 1280

    @classmethod
    def build_vit_b_16( cls ):
        class CustomVisionTransformer(torchvision.models.vision_transformer.VisionTransformer):
            def __init__(self, *args, **kwargs):
                super(CustomVisionTransformer, self).__init__(*args, **kwargs)

            def forward(self, x: torch.Tensor):
                # Reshape and permute the input tensor
                x = self._process_input(x)
                n = x.shape[0]

                print(x.shape)

                # Expand the class token to the full batch
                batch_class_token = self.class_token.expand(n, -1, -1)
                x = torch.cat([batch_class_token, x], dim=1)

                print(x.shape)

                x = self.encoder(x)

                # Classifier "token" as used by standard language architectures
                x = x[:, 0]

                # x = self.heads(x)

                return x.unsqueeze(-1).unsqueeze(-1) # for compativle to other network

        # Download pretrained ViT model
        model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
        pretrained_dict = model.state_dict()

        network = CustomVisionTransformer(
                image_size=224,
                patch_size=16,
                num_layers=12,
                num_heads=12,
                hidden_dim=768,
                mlp_dim=3072,
                )

        network_dict = network.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in network_dict}

        network_dict.update(pretrained_dict)
        network.load_state_dict(network_dict)


        return network, 768



if __name__=='__main__':
    print( BuildNet.available_names() )
