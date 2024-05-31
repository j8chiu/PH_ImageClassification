from doctest import OutputChecker
from pyclbr import Class
from numpy import identity
import torch
import torch.nn as nn
from models.pd_encoder import PersistenceDiagramEncoder, Classifier

class bottleneck(nn.Module):
    def __init__(self, input_channels, output_channels, identity_downsample = None, stride = 1):
        super(bottleneck, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(output_channels)

        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3,stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channels)

        self.conv3 = nn.Conv2d(output_channels, output_channels*self.expansion, kernel_size=1, stride=1,padding=0)
        self.bn3 = nn.BatchNorm2d(output_channels*self.expansion)
        
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        #if identity == None, input shape = (C,H,W) = output shape
        #if identity != None, input shape = (C,H,W), output shape = (4C,H,W) for first stage
        #(2C,H/2,W/2) for the rest stages
        identity = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x
    
class ResNet(nn.Module):
    #block is the bottleneck, stages is like [3,4,6,3], image_channels
    def __init__(self, bottleneck, stages, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.input_channels = 64

        #building stage 0 NN block
        #input shape = (3,H,W) output shape = (64,H/4,W/4)
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #building stage 1 to 4
        #input shape = (64,H/4,W/4) output shape = (256,H/4,W/4)
        self.stage1 = self._make_layer(bottleneck, stages[0], out_channels=64, stride=1)
        #input shape = (256,H/4,W/4) output shape = (512,H/8,W/8)
        self.stage2 = self._make_layer(bottleneck, stages[1], out_channels=128, stride=2)
        #input shape = (512,H/8,W/8) output shape = (1024,H/16,W/16)
        self.stage3 = self._make_layer(bottleneck, stages[2], out_channels=256, stride=2)
        #input shape = (1024,H/16,W/16) output shape = (2048,H/32,W/32)
        self.stage4 = self._make_layer(bottleneck, stages[3], out_channels=512, stride=2)

        #making fc layer
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, num_classes)

        #making fc layer for pd_encoder to match sizes
        self.pd_layer_1 = nn.Sequential(nn.Linear(2048, 256), nn.Sigmoid())
        self.pd_layer_2 = nn.Sequential(nn.Linear(2048, 512), nn.Sigmoid())
        self.pd_layer_3 = nn.Sequential(nn.Linear(2048, 1024), nn.Sigmoid())
        self.pd_layer_4 = nn.Sequential(nn.Linear(2048, 2048), nn.Sigmoid())

        #making pd encoder and persistent homology guided residual block according to paper
        #pd encoder will generate a topological feature vector of 1024 dimensions
        self.pd_encoder = PersistenceDiagramEncoder(input_dim=4)
        self.pd_head = Classifier()
        #add a two layer MLP for transferring topological feature vector to fit persistent homology guided residual block
        self.ph_guided_res_block = nn.Sequential(
            nn.Linear(1024,2048),
            nn.ReLU()
        )

    def _make_layer(self, block, num_res_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        #if it is the first block in a stage, do a downsample  
        if stride != 1 or self.input_channels != 4 * out_channels:
            identity_downsample = nn.Sequential(nn.Conv2d(
                self.input_channels, out_channels * 4, kernel_size = 1, stride = stride
            ),
            nn.BatchNorm2d(out_channels*4))
        #making the first block in a stage
        layers.append(block(self.input_channels, out_channels, identity_downsample, stride))
        #channels being expanded 4 times
        self.input_channels = out_channels*4
        
        #making rest blocks
        #not changing the shape of data
        for i in range(num_res_blocks - 1):
            layers.append(block(self.input_channels, out_channels))
        
        return nn.Sequential(*layers)

    
    def _forward_impl(self,x,topo_feature):
        #input x: a batch of images (batchsize,3,224,224)
        #      topo_feature: a batch of not yet encoded persistent diagrams (batchsize,d,4), where d is not specified
        #output x: resnet result (batchsize,7)
        #       topo_result: (batchsize,7)
        #encoding topo_feature_vector
        topo_feature_vector = self.pd_encoder(topo_feature)
        topo_result = self.pd_head(topo_feature_vector)
        #making persistent homology guided residual block
        topo_res_block_data = self.ph_guided_res_block(topo_feature_vector)
        
        #ResNet stage 0
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        #stage 1, fusing topo features and CNN feature map, shape reserves
        x = self.stage1(x)
        topo_feature_for_stage1 = self.pd_layer_1(topo_res_block_data)
        topo_feature_matrix1 = topo_feature_for_stage1.unsqueeze(2).unsqueeze(3).expand_as(x)
        intermedia_results1 = x * topo_feature_matrix1
        x = x + intermedia_results1

        #stage 2, fusing topo features and CNN feature map, shape reserves
        x = self.stage2(x)
        topo_feature_for_stage2 = self.pd_layer_2(topo_res_block_data)
        topo_feature_matrix2 = topo_feature_for_stage2.unsqueeze(2).unsqueeze(3).expand_as(x)
        intermedia_results2 = x * topo_feature_matrix2
        x = x + intermedia_results2

        #stage 3, fusing topo features and CNN feature map, shape reserves
        x = self.stage3(x)
        topo_feature_for_stage3 = self.pd_layer_3(topo_res_block_data)
        topo_feature_matrix3 = topo_feature_for_stage3.unsqueeze(2).unsqueeze(3).expand_as(x)
        intermedia_results3 = x * topo_feature_matrix3
        x = x + intermedia_results3
        
        #stage 4, fusing topo features and CNN feature map, shape reserves
        x = self.stage4(x)
        topo_feature_for_stage4 = self.pd_layer_4(topo_res_block_data)
        topo_feature_matrix4 = topo_feature_for_stage4.unsqueeze(2).unsqueeze(3).expand_as(x)
        intermedia_results4 = x * topo_feature_matrix4
        x = x + intermedia_results4
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)


        return x, topo_result
    

    def forward(self,x,topo_feature):
        return self._forward_impl(x,topo_feature)
def ResNet50(image_channels = 3, num_classes = 7):
    return ResNet(bottleneck, [3,4,6,3], image_channels, num_classes)

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ResNet50()
    x = torch.randn(16,3,224,224)
    y = torch.randn(16,456,4)
    cv_result, topo_result = net(x,y)
    print(f'resnet result shape {cv_result.shape}')
    print(f'topo result shape {topo_result.shape}')

if __name__ == "__main__":
    test()