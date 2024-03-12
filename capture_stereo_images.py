import cv2
import depthai as dai
import os
import threading
from queue import Queue

# Output directories
output_dir_left = 'left_images'
output_dir_right = 'right_images'
os.makedirs(output_dir_left, exist_ok=True)
os.makedirs(output_dir_right, exist_ok=True)

num_images = 15000  # Total number of images to capture
frame_skip = 16     # Number of frames to skip before saving an image

pipeline = dai.Pipeline()

# Setup mono cameras
left = pipeline.createMonoCamera()
right = pipeline.createMonoCamera()
left.setBoardSocket(dai.CameraBoardSocket.CAM_B)  # Updated to CAM_B
right.setBoardSocket(dai.CameraBoardSocket.CAM_C)  # Updated to CAM_C
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

# Setup XLinkOut
xout_left = pipeline.createXLinkOut()
xout_right = pipeline.createXLinkOut()
xout_left.setStreamName("left")
xout_right.setStreamName("right")

left.out.link(xout_left.input)
right.out.link(xout_right.input)

# Queue for image saving
save_queue = Queue()
stop_signal = threading.Event()

def save_image():
    while not stop_signal.is_set() or not save_queue.empty():
        item = save_queue.get()
        if item is None:
            break
        cv2.imwrite(*item)
        save_queue.task_done()

# Start the saving thread
save_thread = threading.Thread(target=save_image, daemon=True)
save_thread.start()

def main():
    with dai.Device(pipeline) as device:
        leftQueue = device.getOutputQueue(name="left", maxSize=4, blocking=False)
        rightQueue = device.getOutputQueue(name="right", maxSize=4, blocking=False)

        image_count = 0
        frame_counter = 0

        # Initialize display windows in the main thread
        cv2.namedWindow('Left Image')
        cv2.namedWindow('Right Image')

        while image_count < num_images and not stop_signal.is_set():
            inLeft = leftQueue.tryGet()
            inRight = rightQueue.tryGet()

            if inLeft is not None and inRight is not None:
                frame_left = inLeft.getCvFrame()
                frame_right = inRight.getCvFrame()

                # Display the frames
                cv2.imshow('Left Image', frame_left)
                cv2.imshow('Right Image', frame_right)

                if frame_counter % frame_skip == 0:
                    left_filename = os.path.join(output_dir_left, f'left_{image_count:05d}.png')
                    right_filename = os.path.join(output_dir_right, f'right_{image_count:05d}.png')
                    save_queue.put((left_filename, frame_left))
                    save_queue.put((right_filename, frame_right))
                    image_count += 1

                frame_counter += 1

                if cv2.waitKey(1) == ord('q'):
                    stop_signal.set()

if __name__ == "__main__":
    try:
        main()
    finally:
        stop_signal.set()
        save_queue.put(None)  # Signal the save thread to exit
        save_thread.join()
        cv2.destroyAllWindows()
