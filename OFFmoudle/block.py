import torch
import torch.nn as nn
import torch.nn.functional as F
# import spconv.pytorch as spconv
from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, Conv2, DCNV3_conv,autopad,Conv_gn
from .conv import Conv_onany as flow_conv
print("now flow_conv:", flow_conv)
from .transformer import TransformerBlock
from .utils import getPatchFromFullimg,normMask,transform,DLT_solve,homo_align
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load
from .memory_buffer import MutiFeatureBuffer, FeatureBuffer, FlowBuffer
# __all__ = ('DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3', 'C2f', 'C3x', 'C3TR', 'C3Ghost',
#           'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'RepC3')
from torch.cuda.amp import autocast, GradScaler #
from .bsa import BSA
from ultralytics.utils.plotting import overlay_heatmap_on_video
import cv2
from ultralytics.nn.modules.flow import CorrBlock, AlternateCorrBlock, initialize_flow,SepConvGRU,  BasicUpdateBlock, SmallNetUpdateBlock,  NetUpdateBlock, SmallUpdateBlock, warp_feature, ConvGRU
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_




class OFFM(nn.Module): #
    def __init__(self, inchannle, hidden_dim=64, epoch_train=22, stride=[2,2,2], radius=[3,3,3], n_levels=3, iter_max=2, method = "method1", motion_flow = True, aux_loss = False):
        super(OFFM, self).__init__()
        input_dim = inchannle[0] 

        if not (input_dim//2 >= hidden_dim):
            print("************warning****************")
            print(f"input_dim//2 need bigger than hidden_dim, {inchannle},{hidden_dim}") 
            print("***********************************")

        self.inchannle = inchannle

        self.hidden_dim = hidden_dim # input_dim//2
        self.iter_max = iter_max
        self.n_levels = n_levels
        self.radius = radius
        self.stride = stride
        self.epoch_train = epoch_train
        self.method = method
        self.aux_loss = aux_loss
        self.motion_flow = motion_flow
        self.cor_planes = [n_levels * (2*radiu + 1)**2 for radiu in radius]


        self.convs1 = nn.ModuleList([flow_conv(inchannle[i], self.hidden_dim, 1) 
                                    for i in range(n_levels)])
        
        self.convs2 = nn.ModuleList([flow_conv(self.hidden_dim + self.hidden_dim//2, inchannle[i], 1) 
                                    for i in range(n_levels)]) 
        
        # buffer
        self.buffer = FlowBuffer("MemoryAtten", number_feature=n_levels)

        cor_plane = self.cor_planes[1]
        self.cor_plane = cor_plane
        self.cor_plane = 2*(self.cor_plane//2) 

        self.flow_fused0 = FlowUp(self.cor_planes[2], self.cor_planes[1], self.cor_plane)
        self.flow_fused1 = FlowUp(self.cor_plane,self.cor_planes[0], self.cor_plane)
        
        self.flow_fused2 = FlowDown(self.cor_plane ,self.cor_plane, self.cor_planes[2], self.cor_plane)
        
        self.update_block = SmallNetUpdateBlock(input_dim=self.hidden_dim//2, hidden_dim=self.hidden_dim//2, cor_plane = self.cor_plane)

        self.plot = False
        self.save_dir = "./OFFM_saveDir"
        self.pad_image_func = None
        
        

    def forward(self,x):
        # self.plot = True
        with autocast(dtype=torch.float32):
            img_metas = x[0]["img_metas"]
            fmaps_new = x[1:]  #3,B,C,H,W 
            
            if fmaps_new[0].device.type == "cpu" or (img_metas[0]["epoch"] < self.epoch_train and self.training) or self.epoch_train == 100:
                if self.epoch_train == 100:
                    return x[1:]
                outs = []
                for i in range(self.n_levels):
                    out, fmap = self.convs1[i](x[1:][i]).chunk(2, 1)
                    # out, fmap = self.convs0[i](x[1:][i]).chunk(2, 1) #128,128
                    # fmap = self.convs1[i](fmap) #hidden
                    out = self.convs2[i](torch.cat([out,fmap,torch.relu(fmap)], 1))
                    outs.append(out)
                return outs

            
            out_list = []
            fmaps_new = []
            inp = []
            for i in range(self.n_levels):
                out, fmap = self.convs1[i](x[1:][i]).chunk(2, 1)
                # out, fmap = self.convs0[i](x[1:][i]).chunk(2, 1) #128,128
                # fmap = self.convs1[i](fmap) #hidden
                out_list.append(out)
                fmaps_new.append(fmap)
                inp.append(torch.relu(fmap))
                
            if not self.training and self.plot:
                video_name = img_metas[0]["video_name"]
                save_dir = os.path.join(self.save_dir, video_name)
                os.makedirs(save_dir, exist_ok=True)
                image_path = img_metas[0]["image_path"]

                image = cv2.imread(image_path)
                image,_ = self.pad_image_func(image)
                height, width, _ = image.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    
                if img_metas[0]["is_first"]:
                    if hasattr(self, "video_writer_fea"):
                        self.video_writer_fea.release()
                        # self.video_writer_flow.release()
                        # self.video_writer_net.release()
                    self.number = 0
                    self.video_writer_fea = cv2.VideoWriter(os.path.join(save_dir, "feature_fused.mp4"), fourcc, 25, (width, height))
                    # self.video_writer_flow = cv2.VideoWriter(os.path.join(save_dir, "flows.mp4"), fourcc, 25, (width, height))
                    # self.video_writer_net = cv2.VideoWriter(os.path.join(save_dir, "nets.mp4"), fourcc, 25, (width, height))
                # if self.number%600==0:
                #     self.video_writer_fea.release()
                #     self.video_writer_fea = cv2.VideoWriter(os.path.join(save_dir, f"feature_fused_{int(self.number/600)}.mp4"), fourcc, 25, (width, height))

            src_flatten_new, spatial_shapes, level_start_index = self.buffer.flatten(fmaps_new, True)

            result_first_frame, fmaps_old, net_old, coords0, coords1, topk_bbox = self.buffer.update_memory(src_flatten_new, img_metas, spatial_shapes, level_start_index)

            corr_fn_muti = []
            for lvl in range(self.n_levels):
                corr_fn_muti.append(AlternateCorrBlock(fmaps_new[lvl], fmaps_old, self.n_levels, self.radius[lvl], self.stride))

            # 1/32
            lvl = 2
            corr_32 = corr_fn_muti[lvl](coords1[lvl])
            flow_32 = coords1[lvl] - coords0[lvl]
            
            # 1/16
            lvl = 1
            corr_16 = corr_fn_muti[lvl](coords1[lvl])
            flow_16 = coords1[lvl] - coords0[lvl]
            corr_16_fused = self.flow_fused0(corr_32, corr_16)
            net_16, _, delta_flow = self.update_block(net_old[lvl], inp[lvl], corr_16_fused, flow_16)
            coords1[lvl] = coords1[lvl] + delta_flow

            # 1/8
            lvl = 0
            corr_8 = corr_fn_muti[lvl](coords1[lvl])
            flow_8 = coords1[lvl] - coords0[lvl]
            corr_8_fused = self.flow_fused1(corr_16_fused, corr_8)
            net_8, _, delta_flow = self.update_block(net_old[lvl], inp[lvl], corr_8_fused, flow_8)
            coords1[lvl] = coords1[lvl] + delta_flow
            corr_8To32 = F.interpolate(corr_8_fused, scale_factor=1/4, mode="bilinear", align_corners=True)
            corr_16To32 = F.interpolate(corr_16_fused, scale_factor=1/2, mode="bilinear", align_corners=True)

            # 1/32
            lvl = 2
            corr_32_fused = self.flow_fused2(corr_8_fused, corr_16_fused, corr_32)
            net_32, _, delta_flow = self.update_block(net_old[lvl], inp[lvl], corr_32_fused, flow_32)
            coords1[lvl] = coords1[lvl] + delta_flow

            net = [net_8, net_16, net_32]
            self.buffer.update_coords(coords1)
            self.buffer.update_net(net)
                
            for i in range(self.n_levels):
                # if not self.training and self.plot:
                #     np.save(os.path.join(flow_dir, f'level_{i}_{frame_number}.npy'), (coords1[i]-coords0[i]).cpu().numpy())
                #     print(torch.sum(coords1[i]-coords0[i]))
                #     np.save(os.path.join(net_dir, f'level_{i}_{frame_number}.npy'), net[i].cpu().numpy())
                fmaps_new[i] = self.convs2[i](torch.cat([out_list[i],fmaps_new[i],net[i]], 1)) + x[1:][i]
                # if not self.training and self.plot:
                #     np.save(os.path.join(feature_fused_dir,  f'level_{i}_{frame_number}.npy'), fmaps_new[i].cpu().numpy())

            if not self.training and self.plot:
                # overlay_heatmap_on_video(self.video_writer_flow, image, coords1)
                # overlay_heatmap_on_video(self.video_writer_net, image, net)
                overlay_heatmap_on_video(self.video_writer_fea, image, fmaps_new)
                self.number+=1
                
            return fmaps_new
            

class VelocityNet_baseline3_split_dim(OFFM): #
    def __init__(self, inchannle, hidden_dim=64, epoch_train=22, stride=[2,2,2], radius=[3,3,3], n_levels=3, iter_max=2, method = "method1", motion_flow = True, aux_loss = False):
        super().__init__(inchannle=inchannle, hidden_dim=hidden_dim, epoch_train=epoch_train, 
                                          stride=stride, radius=radius, n_levels=n_levels, iter_max=iter_max, 
                                          method=method, motion_flow=motion_flow, aux_loss=aux_loss)
        
        self.dim1 = inchannle[0]-self.hidden_dim
        self.dim2 = self.hidden_dim

        self.convs1 = nn.ModuleList([flow_conv(inchannle[i], inchannle[0], 1) 
                                    for i in range(n_levels)])
        
        self.convs2 = nn.ModuleList([flow_conv(self.dim1 + self.dim2 + self.dim2, inchannle[i], 1) 
                                    for i in range(n_levels)])  # optional act=FReLU(c2)
        
        # self.flow_fused0 = flow_conv(self.cor_planes[2]+ self.cor_planes[1], self.cor_plane, 3, padding=1)
        # self.flow_fused1 = flow_conv(self.cor_planes[0]+ self.cor_plane, self.cor_plane, 3, padding=1)
        # self.flow_fused2 = flow_conv(self.cor_planes[2] + self.cor_plane + self.cor_plane, self.cor_plane, 3, padding=1)
        self.cor_plane = 2*(self.cor_plane//2) #保证是偶数
        self.flow_fused0 = FlowUp(self.cor_planes[2], self.cor_planes[1], self.cor_plane)
        self.flow_fused1 = FlowUp(self.cor_plane,self.cor_planes[0], self.cor_plane)
        
        self.flow_fused2 = FlowDown(self.cor_plane ,self.cor_plane, self.cor_planes[2], self.cor_plane)
        
        self.update_block = SmallNetUpdateBlock(input_dim=self.dim2, hidden_dim=self.dim2, cor_plane = self.cor_plane)

        

    def forward(self,x):
        # self.plot = True
        with autocast(dtype=torch.float32):
            img_metas = x[0]["img_metas"]
            fmaps_new = x[1:]  #3,B,C,H,W 传入的尺度从大到小
            
            if fmaps_new[0].device.type == "cpu" or (img_metas[0]["epoch"] < self.epoch_train and self.training) or self.epoch_train == 100:
                if self.epoch_train == 100:
                    return x[1:], {"k":0.1, "loss":torch.tensor(0.0)}
                outs = []
                for i in range(self.n_levels):
                    out, fmap = self.convs1[i](x[1:][i]).split((self.dim1, self.dim2), 1)
                    # out, fmap = self.convs0[i](x[1:][i]).chunk(2, 1) #128,128
                    # fmap = self.convs1[i](fmap) #hidden
                    out = self.convs2[i](torch.cat([out,fmap,torch.relu(fmap)], 1))
                    outs.append(out)
                return outs, {"k":0.1, "loss":torch.tensor(0.0)}

            
            # 梯度分流,维度一致
            out_list = []
            fmaps_new = []
            inp = []
            for i in range(self.n_levels):
                out, fmap = self.convs1[i](x[1:][i]).split((self.dim1, self.dim2), 1)
                # out, fmap = self.convs0[i](x[1:][i]).chunk(2, 1) #128,128
                # fmap = self.convs1[i](fmap) #hidden
                out_list.append(out)
                fmaps_new.append(fmap)
                inp.append(torch.relu(fmap))
                if not self.training and self.plot:
                    save_dir = "/data/jiahaoguo/ultralytics/runs/UAVTOD_exper/baseline/baseline3/train94_36.2/save_tansor/"
                    video_name = img_metas[0]["video_name"]
                    save_dir = os.path.join(save_dir, video_name)
                    
                    feature_new_dir = os.path.join(save_dir, "feature_new")
                    feature_fused_dir = os.path.join(save_dir, "feature_fused")
                    flow_dir = os.path.join(save_dir, "flows")
                    net_dir = os.path.join(save_dir, "nets")
                    os.makedirs(feature_new_dir, exist_ok=True)
                    os.makedirs(feature_fused_dir, exist_ok=True)
                    os.makedirs(flow_dir, exist_ok=True)
                    os.makedirs(net_dir, exist_ok=True)

                    frame_number = img_metas[0]["frame_number"]
                    np.save(os.path.join(feature_new_dir, f'level_{i}_{frame_number}.npy'), x[1:][i].cpu().numpy())


            src_flatten_new, spatial_shapes, level_start_index = self.buffer.flatten(fmaps_new, True)

            result_first_frame, fmaps_old, net_old, coords0, coords1, topk_bbox = self.buffer.update_memory(src_flatten_new, img_metas, spatial_shapes, level_start_index)

            corr_fn_muti = []
            for lvl in range(self.n_levels):
                corr_fn_muti.append(AlternateCorrBlock(fmaps_new[lvl], fmaps_old, self.n_levels, self.radius[lvl], self.stride))

            # 1/32
            lvl = 2
            corr_32 = corr_fn_muti[lvl](coords1[lvl])
            flow_32 = coords1[lvl] - coords0[lvl]
            
            # 1/16
            lvl = 1
            corr_16 = corr_fn_muti[lvl](coords1[lvl])
            flow_16 = coords1[lvl] - coords0[lvl]
            corr_16_fused = self.flow_fused0(corr_32, corr_16)
            net_16, _, delta_flow = self.update_block(net_old[lvl], inp[lvl], corr_16_fused, flow_16)
            coords1[lvl] = coords1[lvl] + delta_flow

            # 1/8
            lvl = 0
            corr_8 = corr_fn_muti[lvl](coords1[lvl])
            flow_8 = coords1[lvl] - coords0[lvl]
            corr_8_fused = self.flow_fused1(corr_16_fused, corr_8)
            net_8, _, delta_flow = self.update_block(net_old[lvl], inp[lvl], corr_8_fused, flow_8)
            coords1[lvl] = coords1[lvl] + delta_flow
            corr_8To32 = F.interpolate(corr_8_fused, scale_factor=1/4, mode="bilinear", align_corners=True)
            corr_16To32 = F.interpolate(corr_16_fused, scale_factor=1/2, mode="bilinear", align_corners=True)

            # 1/32
            lvl = 2
            corr_32_fused = self.flow_fused2(corr_8_fused, corr_16_fused, corr_32)
            net_32, _, delta_flow = self.update_block(net_old[lvl], inp[lvl], corr_32_fused, flow_32)
            coords1[lvl] = coords1[lvl] + delta_flow

            net = [net_8, net_16, net_32]
            self.buffer.update_coords(coords1)
            self.buffer.update_net(net)


            for i in range(self.n_levels):
                if not self.training and self.plot:
                    np.save(os.path.join(flow_dir, f'level_{i}_{frame_number}.npy'), (coords1[i]-coords0[i]).cpu().numpy())
                    print(torch.sum(coords1[i]-coords0[i]))
                    np.save(os.path.join(net_dir, f'level_{i}_{frame_number}.npy'), net[i].cpu().numpy())
                fmaps_new[i] = self.convs2[i](torch.cat([out_list[i],fmaps_new[i],net[i]], 1)) + x[1:][i]
                if not self.training and self.plot:
                    np.save(os.path.join(feature_fused_dir,  f'level_{i}_{frame_number}.npy'), fmaps_new[i].cpu().numpy())

            return fmaps_new, {"k":0.1, "loss":torch.tensor(0.0).to(net_32.device)}
            

class OFFM_singal_flow(VelocityNet_baseline1): #
    def __init__(self, inchannle, hidden_dim=64, epoch_train=22, stride=[1,1,1], radius=[3,3,3], level_use=[0,1,2], n_levels=3, iter_max=2, method = "method1", motion_flow = True, aux_loss = False):
        super().__init__(inchannle=inchannle, hidden_dim=hidden_dim, epoch_train=epoch_train, 
                                          stride=stride, radius=radius, n_levels=n_levels, iter_max=iter_max, 
                                          method=method, motion_flow=motion_flow, aux_loss=aux_loss)
        
        self.convs1 = nn.ModuleList([flow_conv(inchannle[i], self.hidden_dim, 1) 
                                    for i in range(n_levels)])
        
        self.convs2 = nn.ModuleList([flow_conv(self.hidden_dim + self.hidden_dim//2, inchannle[i], 1) 
                                    for i in range(n_levels)])  # optional act=FReLU(c2)
        
        # self.flow_fused0 = flow_conv(self.cor_planes[2]+ self.cor_planes[1], self.cor_plane, 3, padding=1)
        # self.flow_fused1 = flow_conv(self.cor_planes[0]+ self.cor_plane, self.cor_plane, 3, padding=1)
        # self.flow_fused2 = flow_conv(self.cor_planes[2] + self.cor_plane + self.cor_plane, self.cor_plane, 3, padding=1)
        self.level_use = level_use
        # self.cor_plane = 2*(self.cor_plane//2) 
        self.flow_fused0 = FlowUp(self.cor_planes[2], self.cor_planes[1], self.cor_plane)
        self.flow_fused1 = FlowUp(self.cor_plane,self.cor_planes[0], self.cor_plane)
        
        self.flow_fused2 = FlowDown(self.cor_plane ,self.cor_plane, self.cor_planes[2], self.cor_plane)
        
        self.update_block = SmallNetUpdateBlock(input_dim=self.hidden_dim//2, hidden_dim=self.hidden_dim//2, cor_plane = self.cor_plane)

    def forward(self,x):
        with autocast(dtype=torch.float32):
            img_metas = x[0]["img_metas"]
            fmaps_new = x[1:]  #3,B,C,H,W

            if fmaps_new[0].device.type == "cpu" or (img_metas[0]["epoch"] < self.epoch_train and self.training) or self.epoch_train == 100:
                if self.epoch_train == 100:
                    return x[1:]
                outs = []
                for i in range(self.n_levels):
                    out, fmap = self.convs1[i](x[1:][i]).chunk(2, 1)
                    # out, fmap = self.convs0[i](x[1:][i]).chunk(2, 1) #128,128
                    # fmap = self.convs1[i](fmap) #hidden
                    out = self.convs2[i](torch.cat([out,fmap,torch.relu(fmap)], 1))
                    outs.append(out)
                return outs
            
            out_list = []
            fmaps_new = []
            inp = []
            for i in range(self.n_levels):
                out, fmap = self.convs1[i](x[1:][i]).chunk(2, 1)
                # out, fmap = self.convs0[i](x[1:][i]).chunk(2, 1) #128,128
                # fmap = self.convs1[i](fmap) #hidden
                out_list.append(out)
                fmaps_new.append(fmap)
                inp.append(torch.relu(fmap))
                if not self.training and self.plot:
                    save_dir = "/data/jiahaoguo/ultralytics/runs/save/train243_flow6_2.88/save_tansor/"
                    frame_number = img_metas[0]["frame_number"]
                    np.save(save_dir + f'fmaps_new_{i}_{frame_number}.npy', x[1:][i].cpu().numpy())
                    np.save(save_dir + f'inp_{i}_{frame_number}.npy', inp[i].cpu().numpy())

            src_flatten_new, spatial_shapes, level_start_index = self.buffer.flatten(fmaps_new, True)

            result_first_frame, fmaps_old, net_old, coords0, coords1, topk_bbox = self.buffer.update_memory(src_flatten_new, img_metas, spatial_shapes, level_start_index)

            corr_fn_muti = []
            for lvl in range(self.n_levels):
                corr_fn_muti.append(AlternateCorrBlock(fmaps_new[lvl], fmaps_old, self.n_levels, self.radius[lvl], self.stride))

            # 1/32
            lvl = 2
            if lvl in self.level_use:
                corr_32 = corr_fn_muti[lvl](coords1[lvl])
                flow_32 = coords1[lvl] - coords0[lvl]
            
            # 1/16
            lvl = 1
            if lvl in self.level_use:
                corr_16 = corr_fn_muti[lvl](coords1[lvl])
                flow_16 = coords1[lvl] - coords0[lvl]
                net_16, _, delta_flow = self.update_block(net_old[lvl], inp[lvl], corr_16, flow_16)
                coords1[lvl] = coords1[lvl] + delta_flow
            else:
                net_16 = inp[lvl]

            # 1/8
            lvl = 0
            if lvl in self.level_use:
                corr_8 = corr_fn_muti[lvl](coords1[lvl])
                flow_8 = coords1[lvl] - coords0[lvl]
                net_8, _, delta_flow = self.update_block(net_old[lvl], inp[lvl], corr_8, flow_8)
                coords1[lvl] = coords1[lvl] + delta_flow
            else:
                net_8 = inp[lvl]

            # 1/32
            lvl = 2
            if lvl in self.level_use:
                net_32, _, delta_flow = self.update_block(net_old[lvl], inp[lvl], corr_32, flow_32)
                coords1[lvl] = coords1[lvl] + delta_flow
            else:
                net_32 = inp[lvl]

            #get coords1\net_8\net_16\net_32
            net = [net_8, net_16, net_32]
            self.buffer.update_coords(coords1)
            self.buffer.update_net(net)


            for i in range(self.n_levels):
                if not self.training and self.plot:
                    np.save(save_dir + f'flow{i}_{frame_number}.npy', (coords1[i]-coords0[i]).cpu().numpy())
                    np.save(save_dir + f'net{i}_{frame_number}.npy', net[i].cpu().numpy())
                fmaps_new[i] = self.convs2[i](torch.cat([out_list[i],fmaps_new[i],net[i]], 1)) + x[1:][i]
                if not self.training and self.plot:
                    np.save(save_dir + f'fmaps_new_fused_{i}_{frame_number}.npy', fmaps_new[i].cpu().numpy())

            return fmaps_new
            
class VelocityNet_baseline3_iter(OFFM): #
    def __init__(self, inchannle, hidden_dim=64, epoch_train=22, stride=[2,2,2], radius=[3,3,3], iter_max=2, n_levels=3, method = "method1", motion_flow = True, aux_loss = False):
        super().__init__(inchannle=inchannle, hidden_dim=hidden_dim, epoch_train=epoch_train, 
                                          stride=stride, radius=radius, n_levels=n_levels, iter_max=iter_max, 
                                          method=method, motion_flow=motion_flow, aux_loss=aux_loss)
        self.iter_max = iter_max

    def forward(self,x):
        with autocast(dtype=torch.float32):
            img_metas = x[0]["img_metas"]
            fmaps_new = x[1:]  #3,B,C,H,W 
            
            if fmaps_new[0].device.type == "cpu" or (img_metas[0]["epoch"] < self.epoch_train and self.training) or self.epoch_train == 100:
                if self.epoch_train == 100:
                    return x[1:], {"k":0.1, "loss":torch.tensor(0.0)}
                outs = []
                for i in range(self.n_levels):
                    out, fmap = self.convs1[i](x[1:][i]).chunk(2, 1)
                    # out, fmap = self.convs0[i](x[1:][i]).chunk(2, 1) #128,128
                    # fmap = self.convs1[i](fmap) #hidden
                    out = self.convs2[i](torch.cat([out,fmap,torch.relu(fmap)], 1))
                    outs.append(out)
                return outs, {"k":0.1, "loss":torch.tensor(0.0)}

            out_list = []
            fmaps_new = []
            inp = []
            for i in range(self.n_levels):
                out, fmap = self.convs1[i](x[1:][i]).chunk(2, 1)
                # out, fmap = self.convs0[i](x[1:][i]).chunk(2, 1) #128,128
                # fmap = self.convs1[i](fmap) #hidden
                out_list.append(out)
                fmaps_new.append(fmap)
                inp.append(torch.relu(fmap))
                if not self.training and self.plot:
                    save_dir = "/data/jiahaoguo/ultralytics/runs/save/train243_flow6_2.88/save_tansor/"
                    frame_number = img_metas[0]["frame_number"]
                    np.save(save_dir + f'fmaps_new_{i}_{frame_number}.npy', x[1:][i].cpu().numpy())
                    np.save(save_dir + f'inp_{i}_{frame_number}.npy', inp[i].cpu().numpy())

            src_flatten_new, spatial_shapes, level_start_index = self.buffer.flatten(fmaps_new, True)

            result_first_frame, fmaps_old, net_old, coords0, coords1, topk_bbox = self.buffer.update_memory(src_flatten_new, img_metas, spatial_shapes, level_start_index)

            corr_fn_muti = []
            for lvl in range(self.n_levels):
                corr_fn_muti.append(AlternateCorrBlock(fmaps_new[lvl], fmaps_old, self.n_levels, self.radius[lvl], self.stride))

            for _ in range(self.iter_max):

                for lvl in range(self.n_levels):
                    coords1[lvl] = coords1[lvl].detach()

                # 1/32
                lvl = 2
                corr_32 = corr_fn_muti[lvl](coords1[lvl])
                flow_32 = coords1[lvl] - coords0[lvl]
                
                # 1/16
                lvl = 1
                corr_16 = corr_fn_muti[lvl](coords1[lvl])
                flow_16 = coords1[lvl] - coords0[lvl]
                corr_16_fused = self.flow_fused0(corr_32, corr_16)
                net_16, _, delta_flow = self.update_block(net_old[lvl], inp[lvl], corr_16_fused.contiguous(), flow_16)
                coords1[lvl] = coords1[lvl] + delta_flow

                # 1/8
                lvl = 0
                corr_8 = corr_fn_muti[lvl](coords1[lvl])
                flow_8 = coords1[lvl] - coords0[lvl]
                corr_8_fused = self.flow_fused1(corr_16_fused, corr_8)
                net_8, _, delta_flow = self.update_block(net_old[lvl], inp[lvl], corr_8_fused.contiguous(), flow_8)
                coords1[lvl] = coords1[lvl] + delta_flow

                # 1/32
                lvl = 2
                corr_32_fused = self.flow_fused2(corr_8_fused, corr_16_fused, corr_32)
                net_32, _, delta_flow = self.update_block(net_old[lvl], inp[lvl], corr_32_fused.contiguous(), flow_32)
                coords1[lvl] = coords1[lvl] + delta_flow

                inp = [net_8.contiguous(), net_16.contiguous(), net_32.contiguous()]

            #get coords1\net_8\net_16\net_32
            self.buffer.update_coords(coords1)
            self.buffer.update_net(inp)


            for i in range(self.n_levels):
                if not self.training and self.plot:
                    np.save(save_dir + f'flow{i}_{frame_number}.npy', (coords1[i]-coords0[i]).cpu().numpy())
                    np.save(save_dir + f'net{i}_{frame_number}.npy', inp[i].cpu().numpy())
                fmaps_new[i] = self.convs2[i](torch.cat([out_list[i],fmaps_new[i],inp[i]], 1)) + x[1:][i]
                if not self.training and self.plot:
                    np.save(save_dir + f'fmaps_new_fused_{i}_{frame_number}.npy', fmaps_new[i].cpu().numpy())

            return fmaps_new, {"k":0.1, "loss":torch.tensor(0.0).to(net_32.device)}
