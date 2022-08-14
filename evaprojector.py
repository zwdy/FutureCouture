import tensorflow as tf
import numpy as np
import dnnlib
import dnnlib.tflib as tflib

import projector
import pretrained_networks
from training import dataset
from training import misc
import dataset_tool
import PIL.Image
import math

import os
import json
import uuid
import falcon
from falcon_multipart.middleware import MultipartMiddleware

import shutil

GLOBAL_IMAGE_DIR = './images'
if not os.path.exists(GLOBAL_IMAGE_DIR):
    os.mkdir(GLOBAL_IMAGE_DIR)



class PostResource:
    def on_post(self, req, resp):
        image = req.get_param('file')
        styleId = req.get_param('styleid')

        raw = image.file.read()
        folder_id = str(uuid.uuid4())
        upload_dir = os.path.join(GLOBAL_IMAGE_DIR, f'{folder_id}')
        os.mkdir(upload_dir)
        filepath = os.path.join(upload_dir, 'input.png')
        print(filepath)#####this is the file you want to project
        print(f'{styleId}')
        with open(filepath, 'wb') as fp:
            fp.write(raw)
        stylepath = os.path.join(upload_dir, 'style.png')
        shutil.copy(f'./styles/{styleId}',stylepath)


        process_input(upload_dir)


        # you have the uploaded image location "filepath"
        # project that image and put it in the OUTPUT_DIR

        resp.status = falcon.HTTP_200
        resp.content_type = 'application/json'
        resp.body = json.dumps({'folder_id': folder_id})#??revise: change id: see script.js line16 and line43

app = falcon.App(cors_enable=True, middleware=[MultipartMiddleware()])
app.add_route('/submit', PostResource())
#app.add_route('/images/{image_id}', GetResource())



def process_input(inImageDire):



    OUTPUT_DIR = os.path.join(inImageDire, 'generated')
    os.mkdir(OUTPUT_DIR)

    RECORDS_DIR = os.path.join(inImageDire, 'records')
    os.mkdir(RECORDS_DIR)



    ###################################################################################################
    DIR = OUTPUT_DIR #output folder name
    tflib.init_tf()
    #network_pkl = "network-snapshot-010114.pkl" #010108 FIRST ITERATION OF SCULPTURE2RUNWAY
    network_pkl = "D:/Work/SCIArc/2022Summer_Web/pyscript/checkpt-influencer2showlook-015004.pkl"

    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    ###################################################################################################

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    Gs_kwargs.truncation_psi = .5
    trunc_psi=.5
    Z_SIZE = Gs.input_shape[1]
    imgCnt = 2
    ###################################################################################################
    # eva projector image folder： projection/imgs
    print(RECORDS_DIR)
    print(inImageDire)
    dataset_tool.create_from_images_raw(RECORDS_DIR, inImageDire, False)
    latentList = project_real_images(Gs, network_pkl,"records", inImageDire, imgCnt, 1)

    ###################################################################################################
    testImg = Gs.components.synthesis.run(latentList[0], **Gs_kwargs)[0]
    img = PIL.Image.fromarray(testImg, 'RGB')
    img.save("test.png")

    #step is framerate of final video  
    #step = 24 
    #latentStep is the number of frames between each of your projected images
    latentStep = 1
    
       
    
    curPos = 0
    curLatent = 0
    frame_List = []
    for j in range(imgCnt-1):
        latentStart = latentList[j]
        latentEnd = latentList[j+1]
        for i in range(latentStep):
            myCount = i + j*latentStep
            print(i + (j*latentStep))
            factor = i/latentStep            
            current_latent = latentStart*(1-factor) + latentEnd*(factor)
            current_image = Gs.components.synthesis.run(current_latent, **Gs_kwargs)[0]

            frame_List.append(current_image)
            image = PIL.Image.fromarray(current_image, 'RGB')
            cntStr = str(myCount)
            fName = DIR + '/'+'out_' + cntStr + '.png'
            image.save(fName)

#import moviepy.editor



def project_image(proj, targets, png_prefix, num_snapshots):
    snapshot_steps = set(proj.num_steps - np.linspace(0, proj.num_steps, num_snapshots, endpoint=False, dtype=int))    
    proj.start(targets)
    while proj.get_cur_step() < proj.num_steps:
        print('\r%d / %d ... ' % (proj.get_cur_step(), proj.num_steps), end='', flush=True)
        proj.step()
        
    return proj.get_dlatents()
    print('\r%-30s\r' % '', end='', flush=True)
        
def project_real_images(Gs,network_pkl, dataset_name, data_dir, num_images, num_snapshots):
    #print('Loading networks from "%s"...' % network_pkl)
   
    proj = projector.Projector()
    proj.set_network(Gs)
    proj.num_steps = 500

    print('Loading images from "%s"...' % dataset_name)
    dataset_obj = dataset.load_dataset(data_dir=data_dir, tfrecord_dir=dataset_name, max_label_size=0, repeat=False, shuffle_mb=0)
    assert dataset_obj.shape == Gs.output_shape[1:]
    
    latList = []
    for image_idx in range(num_images):
        print('Projecting image %d/%d ...' % (image_idx, num_images))
        images, _labels = dataset_obj.get_minibatch_np(1)
        images = misc.adjust_dynamic_range(images, [0, 255], [-1, 1])
        dlats = project_image(proj, targets=images, png_prefix=dnnlib.make_run_dir_path('image%04d-' % image_idx), num_snapshots=num_snapshots)
        latList.append(dlats)
    return latList

def main():
    ###################################################################################################
    DIR = OUTPUT_DIR #output folder name
    tflib.init_tf()
    #network_pkl = "network-snapshot-010114.pkl" #010108 FIRST ITERATION OF SCULPTURE2RUNWAY
    network_pkl = "checkpt-influencer2showlook-015004.pkl"
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    ###################################################################################################

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False    
    Gs_kwargs.truncation_psi = .9
    trunc_psi=.9
    Z_SIZE = Gs.input_shape[1]
    imgCnt = 2
    ###################################################################################################
    # eva projector image folder： projection/imgs
    dataset_tool.create_from_images_raw("projection/records/",filepath, False)
    latentList = project_real_images(Gs,network_pkl,"records",filepath,imgCnt,1)
    ###################################################################################################
    testImg = Gs.components.synthesis.run(latentList[0], **Gs_kwargs)[0]
    img = PIL.Image.fromarray(testImg, 'RGB')
    img.save("test.png")

    #step is framerate of final video  
    #step = 24 
    #latentStep is the number of frames between each of your projected images
    latentStep = 48
    
       
    
    curPos = 0
    curLatent = 0
    frame_List = []
    for j in range(imgCnt-1):
        latentStart = latentList[j]
        latentEnd = latentList[j+1]
        for i in range(latentStep):
            myCount = i + j*latentStep
            print(i + (j*latentStep))
            factor = i/latentStep            
            current_latent = latentStart*(1-factor) + latentEnd*(factor)
            current_image = Gs.components.synthesis.run(current_latent, **Gs_kwargs)[0]

            frame_List.append(current_image)
            image = PIL.Image.fromarray(current_image, 'RGB')
            cntStr = str(myCount)
            fName = DIR + '/'+'out_' + cntStr + '.png'
            image.save(fName)
            
    #mp4_file = 'influencer2showlook_015004_03.mp4'
    #mp4_codec = 'libx264'
    #mp4_bitrate = '3M'
    #mp4_fps = step

    #frames = moviepy.editor.ImageSequenceClip(frame_List, fps=mp4_fps)
    #frames.write_videofile(mp4_file,fps=mp4_fps,codec=mp4_codec,bitrate=mp4_bitrate)

if __name__ == "__main__":
    main()
