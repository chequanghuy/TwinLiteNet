    
import onnxruntime
import argparse
import numpy as np
from PIL import Image
import os
import cv2
import cv2
import numpy as np
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser(description='Process some images')
    parser.add_argument('--image', type=str, default=None,
                        help='input image')
    parser.add_argument("--video", type=str, default=None, help="input video")
    parser.add_argument('--model', type=str, default='quantized_model_float16.tflite')
    parser.add_argument("--skip-frames", type=int, default=0, help="skip frames")

    args = parser.parse_args()
    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    # Get input and output tensors information
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("input_details",input_details)
    print("output_details",output_details)
    # Load and preprocess an image using Pillow
    #img = Image.open(args.image)
    #img = img.resize(( 640,360))  # Resize the image to match the model's input size
    if args.image is not None:
    # Load an preprocess an image using OpenCV
        img = cv2.imread(args.image)
        assert isinstance(img, np.ndarray), "Fail to read image: {}".format(args.image)
        input_shape = input_details[0]["shape"]
        img_rs = find_lanes(interpreter, input_shape, img, output_details= output_details, input_details=input_details)
        os.makedirs('results',exist_ok=True)
        cv2.imwrite(os.path.join('results',os.path.basename(args.image)),img_rs)
    if args.video:
        cap = cv2.VideoCapture(args.video)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if args.skip_frames > 0:
                for i in range(args.skip_frames):
                    ret, frame = cap.read()
                    if not ret:
                        break
            input_shape = input_details[0]["shape"]
            frame_rs  = find_lanes(interpreter, input_shape, frame, output_details= output_details, input_details=input_details)
            cv2.imshow('frame',frame_rs)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def find_lanes(interpreter, input_shape, img, output_details, input_details):
    img = cv2.resize(img, (input_shape[2],input_shape[1]))

    # Convert the Pillow image to a NumPy array and preprocess
    img_array = img.copy()
    #img_array = img_array[:, :, ::-1].transpose(2, 0, 1)
    #img_array = np.ascontiguousarray(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype(np.float32)  / 255.0  # Normalize pixel values to [0, 1]

    # Shape is 1,3,360,640

    #img_array = img_array.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]
    #img_array = np.transpose(img_array, (2, 0, 1))  # Change the data layout to CHW
    #img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
    #img_array = np.ascontiguousarray(img_array)
    # Perform inference using the ONNX model
    #input_name = interpreter.get_inputs()[0].name
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(0), interpreter.get_tensor(1)
    # Post-process the model's output (adjust based on your model's output format)
    #outputs
    x0, x1 = output_data
    x0 = interpreter.get_tensor(output_details[0]['index'])
    x1 = interpreter.get_tensor(output_details[1]['index'])

    x0 = x0.transpose(0, 3, 1, 2)
    x1 = x1.transpose(0, 3, 1, 2)
    #x0=img_out[0] # torch.Size([1, 2, 360, 640])
    #x1=img_out[1] # torch.Size([1, 2, 360, 640])

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
    img_rs[DA>50]=[255,0,0]
    img_rs[LL>50]=[0,255,0]
    return img_rs

# Start main if called directly
if __name__ == "__main__":
    main()

