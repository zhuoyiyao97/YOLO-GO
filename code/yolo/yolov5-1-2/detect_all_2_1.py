import argparse
import os
import glob
from PIL import Image
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized

def generate_test(DO):
    if DO:
        file = "inference/images/test"
        a = os.path.exists(file)
        if a == False: # 不存在則創建文件夾
            os.makedirs(file)
        else: # 存在则删除然后创建
            for fileList in os.walk(file):
                for name in fileList[2]:
                    os.chmod(os.path.join(fileList[0], name), stat.S_IWRITE)
                    os.remove(os.path.join(fileList[0], name))
            shutil.rmtree(file)
            os.makedirs(file)
        for line in open("data/test.txt"):
            name = line[12:22]
            img = cv2.imread(line[:-1], 0)  # 0：读入的为灰度图像 1：读入的为彩色图像
            cv2.imwrite('inference/images/test/' + name, img)
    else:
        return 0

def detect(save_img=False):
    src_txt_dir = "F:/ZYY/chess/images/test2/tmp/txt-2-1/"

    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        fileName = path.split('\\')[-1].split('.')[0] + '.txt'
        save_f = src_txt_dir + fileName
        file = open(save_f, 'w')
        im = Image.open(path)
        width, height = im.size
        if width > 1200 :
            thickness = 3
        else:
            thickness = 1

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        # result1 = pred.data.cpu().numpy().reshape(-1, 8)
        # np.savetxt('pred1.txt', result1)

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()
        # with open('pred2.txt', 'w') as f:
            # f.write(str(pred))

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            # txt_path = 'F:/FH/FxH'
            # txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                save_txt = True
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # with open(txt_path + '.txt', 'a') as f:
                        #     f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                        label = '%s ' % (names[int(cls)])
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=thickness)

                        string = "corner "
                        if label == string:
                            label = '3'
                            bnd = torch.tensor(xyxy).view(1, 4).numpy()[0]
                            label_bnd = label + ' ' + str(int(bnd[0])) + ' ' + str(int(bnd[1])) + ' ' + str(
                                int(bnd[2])) + ' ' + str(int(bnd[3])) + ' %.2f' % (conf) + '\n'
                            file.write(label_bnd)
                        else:
                            # 1*2的框拆成两个1*1的框
                            # (xmin, ymin, xmax, ymax)
                            bnd = torch.tensor(xyxy).view(1, 4).numpy()[0]
                            label_bnd = label[0] + ' ' + str(int(bnd[0])) + ' ' + str(int(bnd[1])) + ' ' + str(
                                int(bnd[2])) + ' ' + str(int(0.5*(bnd[1] + bnd[3]))) + ' %.2f' % (conf) + '\n'
                            file.write(label_bnd)
                            label_bnd = label[1] + ' ' + str(int(bnd[0])) + ' ' +  str(int(0.5*(bnd[1] + bnd[3]))) + ' ' + str(int(bnd[2])) + ' ' + str(int(bnd[3])) + ' %.2f' % (conf) + '\n'
                            file.write(label_bnd)






            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))

    print('Done. (%.3fs)' % (time.time() - t0))

def generate(src_img_dir, src_txt_dir, src_xml_dir):
    img_Lists = glob.glob(src_img_dir + '/*.jpg')

    img_basenames = []  # e.g. 100.jpg
    for item in img_Lists:
        img_basenames.append(os.path.basename(item))

    img_names = []  # e.g. 100
    for item in img_basenames:
        temp1, temp2 = os.path.splitext(item)
        img_names.append(temp1)

    for img in img_names:
        print(img)
        f = src_img_dir + '/' + img + '.jpg'
        im = Image.open(f)
        width, height = im.size

        # open the crospronding txt file
        gt = open(src_txt_dir + '/' + img + '.txt').read().splitlines()
        # gt_width, gt_height = gt.size
        xml_file = open((src_xml_dir + img + '.xml'), 'w')
        xml_file.write('<annotation>\n')
        xml_file.write('    <folder>test</folder>\n')
        xml_file.write('    <filename>' + str(img) + '.jpg' + '</filename>\n')
        xml_file.write('    <path>' + src_img_dir + '/' + str(img) + '.jpg' + '</path>\n')
        xml_file.write('    <size>\n')
        xml_file.write('        <width>' + str(width) + '</width>\n')
        xml_file.write('        <height>' + str(height) + '</height>\n')
        xml_file.write('        <depth>3</depth>\n')
        xml_file.write('    </size>\n')

        # write the region of image on xml file
        i = 1
        for img_each_label in gt:
            spt = img_each_label.split(' ')  # 这里如果txt里面是以逗号‘，’隔开的，那么就改为spt = img_each_label.split(',')。
            imax = len(spt)
            xml_file.write('    <object>\n')
            xml_file.write('        <name>' + spt[0] + '</name>\n')
            xml_file.write('        <pose>Unspecified</pose>\n')
            xml_file.write('        <truncated>0</truncated>\n')
            xml_file.write('        <difficult>0</difficult>\n')
            xml_file.write('        <bndbox>\n')
            xml_file.write('            <xmin>' + str(max(0, int(spt[1]))) + '</xmin>\n')
            xml_file.write('            <ymin>' + str(max(0, int(spt[2]))) + '</ymin>\n')
            xml_file.write('            <xmax>' + str(min(width, int(spt[3]))) + '</xmax>\n')
            xml_file.write('            <ymax>' + str(min(height, int(spt[4]))) + '</ymax>\n')
            xml_file.write('        </bndbox>\n')
            xml_file.write('    </object>\n')
        xml_file.write('</annotation>')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 给定模型，需要推理的图片路径，输出文件地址即可
    # source = "F:/ZYY/chess/model_ensemble/data/images/train/"
    # out = "F:/ZYY/chess/model_ensemble/data/train/train-2-1-out/"
    # source = "F:/ZYY/chess/model_ensemble/data/images/test/"
    # out = "F:/ZYY/chess/model_ensemble/data/test/test-2-1-out/"

    source = "F:/ZYY/chess/images/test2/tmp/images/"
    out = "F:/ZYY/chess/images/test2/tmp/out-2-1/"

    parser.add_argument('--weights', nargs='+', type=str, default='runs/pt/2-1-x.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=source, help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default=out, help='output folder')  # output folder
    # parser.add_argument('--weights', nargs='+', type=str, default='yolov5l.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.3, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    generate_test(False)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
            os.system("python  VOCxml.py")
            # 注意是append,重新写入txt时需要先删除
            # 图像存储位置
            src_img_dir = "F:/ZYY/chess/yolov5/yolov5-1-2/inference/images/train/"
            # 图像的 ground truth 的 txt 文件存放位置
            src_txt_dir = "F:/ZYY/chess/yolov5/yolov5-1-2/inference/output/test-1-2-txt/"
            src_xml_dir = "F:/ZYY/chess/yolov5/yolov5-1-2/inference/output/test-1-2-xml/"
            # generate(src_img_dir, src_txt_dir, src_xml_dir)