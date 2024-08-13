    
import onnxruntime
import argparse
import numpy as np
from PIL import Image
import os
import cv2
def main():
    parser = argparse.ArgumentParser(description='Process some images')
    parser.add_argument('--image', type=str, default=None,
                        help='input image')
    parser.add_argument("--video", type=str, default=None, help="input video", nargs='+')
    parser.add_argument('--model', type=str, default='twinlitenet.onnx')
    parser.add_argument('--save-video', type=str, default='output.mp4')

    args = parser.parse_args()

    ort_session = onnxruntime.InferenceSession(args.model, providers=["CUDAExecutionProvider"])

    # get input shape
    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape
    print("input_name",input_name)
    print("input_shape",input_shape)
    # Load and preprocess an image using Pillow
    #img = Image.open(args.image)
    #img = img.resize(( 640,360))  # Resize the image to match the model's input size
    if args.image is not None:
    # Load an preprocess an image using OpenCV
        img = cv2.imread(args.image)
        img_rs = find_lanes(ort_session, input_shape, img)
        os.makedirs('results',exist_ok=True)
        cv2.imwrite(os.path.join('results',os.path.basename(args.image)),img_rs)
    elif args.video and len(args.video) > 0:
        videos = args.video
        if args.save_video:
            os.makedirs('output_frames',exist_ok=True)

        frame_number = 0
        for video in videos:
            cap = cv2.VideoCapture(video)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rs = find_lanes(ort_session, input_shape, frame)
                cv2.imshow('frame',frame_rs)
                if args.save_video:
                    cv2.imwrite(os.path.join('output_frames',f'{frame_number:06d}.png'),frame_rs)
                    frame_number += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    else:
        print("No input image or video specified")

def find_lanes(ort_session, input_shape, img):
    img = cv2.resize(img, (input_shape[3],input_shape[2]))

    # Convert the Pillow image to a NumPy array and preprocess
    img_array = img.copy()
    img_array = img_array[:, :, ::-1].transpose(2, 0, 1)
    img_array = np.ascontiguousarray(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]

    # Shape is 1,3,360,640

    #img_array = img_array.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]
    #img_array = np.transpose(img_array, (2, 0, 1))  # Change the data layout to CHW
    #img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
    #img_array = np.ascontiguousarray(img_array)
    # Perform inference using the ONNX model
    input_name = ort_session.get_inputs()[0].name
    img_out = ort_session.run(None, {input_name: img_array}) 

    # Post-process the model's output (adjust based on your model's output format)
    #outputs



    x0=img_out[0] # torch.Size([1, 2, 360, 640])
    x1=img_out[1] # torch.Size([1, 2, 360, 640])

    #max_values_0 = np.max(x0, axis=1)
    da_predict = np.argmax(x0, axis=1)

    #max_values_x1 = np.max(x1, axis=1)
    ll_predict = np.argmax(x1, axis=1)
    #da_predict=np.max(x0, 1) #torch.Size([1, 360, 640])
    #ll_predict=np.max(x1, 1) #torch.Size([1, 360, 640])
    # DA = da_predict.byte().cpu().data.numpy()[0]*255
    # LL = ll_predict.byte().cpu().data.numpy()[0]*255
    DA = da_predict.astype(np.uint8)[0]*255
    LL = ll_predict.astype(np.uint8)[0]*255
    img_rs=img.copy()
    img_rs[DA>50]+=np.array([64,0,0],dtype=np.uint8)
    img_rs[LL>50]+=np.array([0,64,0],dtype=np.uint8)
    return img_rs

# Start main if called directly
if __name__ == "__main__":
    main()

