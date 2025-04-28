import cv2
import os

# --- Configuration ---
video_path = 'justicebuys-home-gadgets.mp4' 

# Dictionary of product titles and their timestamps (in seconds)
products = {
    "15_Lamp_Water_Dispenser": 3,
    "14_Overbed_Table": 7,
    "13_AC_Bedsheet": 10,
    "12_Flower_Frame_Vase": 14,
    "11_Magnet_Door_Stopper": 18,
    "10_Stepping_Stones": 22,
    "09_Laundry_Apron": 26,
    "08_Cherry_Toilet_Scrubber": 29,
    "07_Bird_Feeder_Camera": 33,
    "06_Wall_Hole_Putty": 37,
    "05_Fridge_Shelf_Magnets": 41,
    "04_Fridge_Drink_Dispenser": 45,
    "03_Bedsure_Cooling_Mattress_Pad": 49,
    "02_Secret_Ottoman": 53,
    "01_Slapp_Shop_Bath_Mat": 57
}

# Create output directory if it doesn't exist
output_dir = 'product_images'
os.makedirs(output_dir, exist_ok=True)

# Check if video file exists
if not os.path.exists(video_path):
    print(f"Error: Video file not found at '{video_path}'")
else:
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
    else:
        # Get the frames per second (FPS) of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            print("Error: Could not determine video FPS.")
        else:
            print(f"Video FPS: {fps:.2f}")
            
            # Process each product
            for product_title, target_time_sec in products.items():
                # Calculate the target frame number
                target_frame_number = int(target_time_sec * fps)
                print(f"\nProcessing: {product_title}")
                print(f"Target time: {target_time_sec:.2f} seconds")
                print(f"Calculated target frame: {target_frame_number}")

                # Set the video position to the target frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_number)

                # Read the frame at the specified position
                ret, frame = cap.read()

                if ret:
                    # Create output filename with product title
                    output_filename = os.path.join(output_dir, f"{product_title}.jpg")
                    
                    # Save the extracted frame as an image
                    cv2.imwrite(output_filename, frame)
                    print(f"Successfully extracted and saved frame to '{output_filename}'")
                else:
                    # Check if the time was simply beyond the video length
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = total_frames / fps if fps > 0 else 0
                    if target_frame_number >= total_frames:
                        print(f"Error: Target time ({target_time_sec:.2f}s) is beyond the video duration ({duration:.2f}s).")
                    else:
                        print(f"Error: Could not read frame {target_frame_number}. Seek might be inaccurate or file corrupted.")

            # Release the video capture object
            cap.release()

            print("\nProcessing complete!")

        # Optional: Destroy any OpenCV windows if they were used (not needed here)
        # cv2.destroyAllWindows()