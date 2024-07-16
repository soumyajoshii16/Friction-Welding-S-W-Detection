
# Friction Welding Shaft & Wheel Detection Using CVML

This project addresses the challenge of manual verification of wheel and shaft pairings in friction welding by implementing an automated system using computer vision and machine learning technologies.
1. Camera Integration: Utilizing a strategically positioned camera to capture images of wheel and shaft assemblies before friction welding.
2. Image Pre-processing: Implementing software to pre-process captured images for color detection, enhancing clarity, adjusting color levels, and reducing noise to optimize images for accurate color classification and validation.
3. Deep Learning Model: Incorporating a pre-trained Faster R-CNN (Region-based Convolutional Neural Network) algorithm to efficiently detect and classify wheel types and shaft colours in the pre-processed images.The choice of Faster R-CNN was motivated by its robust feature extraction capabilities and proven effectiveness in spatial localization tasks, making it suitable for accurately identifying and validating complex color patterns and object configurations in industrial settings.
4. Raspberry Pi and PLC Integration: Connecting Raspberry Pi to a Programmable Logic Controller (PLC) to automate machine control based on detected wheel and shaft pairings. The system uses input from the Raspberry Pi to autonomously start or stop the machine, ensuring correct matches for friction welding operations.
5. Output Generation: Generating real-time feedback indicating the validity of the wheel and shaft pairing, ensuring correct matches for friction welding operations.


## Features

1. Wheel Type Selection: Users can choose the specific wheel type they are working with, setting the context for the validation process.
2. Shaft Positioning: Users position the shaft in alignment with the selected wheel type using the webcam feed displayed on the website.
3. Frame Capture: Users trigger the frame capture by pressing the space button on the keyboard, capturing an image of the shaft-wheel assembly for analysis.
4. Real-time Analysis: The captured image is processed within 2 seconds using computer vision and machine learning algorithms to determine the validity of the shaft alignment.
5. Validation Feedback: Immediate feedback is provided to the user, indicating whether the shaft alignment is valid or invalid based on predefined criteria and color detection algorithms.
6. Responsive Design: The website is designed to be responsive, ensuring optimal performance and usability across different devices and screen sizes.
7. Performance Optimization: Emphasis on optimizing processing speed and efficiency to meet the 2-second response time requirement for real-time feedback.


![Screenshot 2024-07-16 135859](https://github.com/user-attachments/assets/6ab6e0f8-3141-47e6-ae9d-3db4b395e5b7)


![type1_frame_20240626_112447](https://github.com/user-attachments/assets/8832de28-17f8-4da0-9578-55d3d9bd895f)


![type2_frame_20240620_155250](https://github.com/user-attachments/assets/c4982ca6-2f02-4a45-992e-ae7f8651cbe3)


