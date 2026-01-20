### normal force record ###
import csv
import datetime
###########################

import cv2
import numpy as np
from config import ConfigModel
from utilities.image_processing import (
    stack_label_above_image,
    apply_cmap,
    color_map_from_txt,
    normalize_array,
    trim_outliers,
)
from utilities.reconstruction import Reconstruction3D
from utilities.visualization import Visualize3D
from utilities.gelsightmini import GelSightMini
from utilities.logger import log_message


import os
record_path = "./recordings/normal_force_1"
if not os.path.exists("./recordings/normal_force_1"):
    os.makedirs("./recordings/normal_force_1")

def UpdateView(
    image: np.ndarray,
    cam_stream: GelSightMini,
    reconstruction: Reconstruction3D,
    visualizer3D: Visualize3D,
    cmap: np.ndarray,
    config: ConfigModel,
    window_title: str,
    # === normal force record ===
    is_logging: bool = False,
    csv_writer=None,
    csv_file=None,
    #############################
):

    # Compute depth map and gradients.
    depth_map, contact_mask, grad_x, grad_y = reconstruction.get_depthmap(
        image=image,
        markers_threshold=(config.marker_mask_min, config.marker_mask_max),
    )

    if visualizer3D:
        visualizer3D.update(depth_map, gradient_x=grad_x, gradient_y=grad_y)

    if np.isnan(depth_map).any():
        return

    depth_map_trimmed = trim_outliers(depth_map, 1, 99)
    depth_map_normalized = normalize_array(array=depth_map_trimmed, min_divider=10)
    depth_rgb = apply_cmap(data=depth_map_normalized, cmap=cmap)

    contact_mask_bool = contact_mask.astype(bool)

    # === depth sum positive (inside contact mask) + CSV logging ===
    depth_sum_pos = float(np.sum(depth_map[(depth_map > 0)]))
    # print("depth sum positive :", depth_sum_pos)

    if is_logging and (csv_writer is not None):
        ts_str = datetime.datetime.now().isoformat(timespec="milliseconds")
        csv_writer.writerow([ts_str, f"{depth_sum_pos:.6f}"])
        if csv_file is not None:
            csv_file.flush()
    # ============================================================

    # Convert masks to 8-bit grayscale.
    contact_mask_u8 = (contact_mask_bool * 255).astype(np.uint8)

    # Convert grayscale images to 3-channel for stacking.
    contact_mask_rgb = cv2.cvtColor(contact_mask_u8, cv2.COLOR_GRAY2BGR)

    # Apply labels above images
    frame_labeled = stack_label_above_image(
        image, f"Camera Feed {int(cam_stream.fps)} FPS", 30
    )

    depth_labeled = stack_label_above_image(depth_rgb, "Depth", 30)
    contact_mask_labeled = stack_label_above_image(contact_mask_rgb, "Contact Mask", 30)

    # Increase spacing between images by adding black spacers
    spacing_size = 30
    horizontal_spacer = np.zeros(
        (frame_labeled.shape[0], spacing_size, 3), dtype=np.uint8
    )

    top_row = np.hstack(
        (
            frame_labeled,
            horizontal_spacer,
            contact_mask_labeled,
            horizontal_spacer,
            depth_labeled,
        )
    )

    display_frame = top_row

    # Scale the display frame
    display_frame = cv2.resize(
        display_frame,
        (
            int(display_frame.shape[1] * config.cv_image_stack_scale),
            int(display_frame.shape[0] * config.cv_image_stack_scale),
        ),
        interpolation=cv2.INTER_NEAREST,
    )
    display_frame = display_frame.astype(np.uint8)

    # Show the combined image.
    cv2.imshow(window_title, display_frame)


def View3D(config: ConfigModel):
    WINDOW_TITLE = "Multi-View (Camera, Contact, Depth)"

    reconstruction = Reconstruction3D(
        image_width=config.camera_width,
        image_height=config.camera_height,
        use_gpu=config.use_gpu,  # Change to True if you want to use CUDA.
    )

    # Load the trained network using the existing method in reconstruction.py.
    if reconstruction.load_nn(config.nn_model_path) is None:
        log_message("Failed to load model. Exiting.")
        return

    if config.pointcloud_enabled:
        # Initialize the 3D Visualizer.
        visualizer3D = Visualize3D(
            pointcloud_size_x=config.camera_width,
            pointcloud_size_y=config.camera_height,
            save_path="",  # Provide a path if you want to save point clouds.
            window_width=int(config.pointcloud_window_scale * config.camera_width),
            window_height=int(config.pointcloud_window_scale * config.camera_height),
        )
    else:
        visualizer3D = None

    cmap = color_map_from_txt(
        path=config.cmap_txt_path, is_bgr=config.cmap_in_BGR_format
    )

    # Initialize the camera stream.
    cam_stream = GelSightMini(
        target_width=config.camera_width, target_height=config.camera_height
    )
    devices = cam_stream.get_device_list()
    log_message(f"Available camera devices: {devices}")
    cam_stream.select_device(config.default_camera_index)
    cam_stream.start()

    # === normal force record: logging state/handles ===
    is_logging = False
    csv_file = None
    csv_writer = None
    csv_path = None
    # ==================================================

    # Main loop: capture frames, compute depth map, and update the 3D view.
    try:
        while True:
            # Get a new frame from the camera.
            frame = cam_stream.update(dt=0)
            if frame is None:
                continue

            # Convert color
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            UpdateView(
                image=frame,
                cam_stream=cam_stream,
                reconstruction=reconstruction,
                visualizer3D=visualizer3D,
                cmap=cmap,
                config=config,
                window_title=WINDOW_TITLE,
                # === normal force record args ===
                is_logging=is_logging,
                csv_writer=csv_writer,
                csv_file=csv_file,
                ##############################
            )

            key = cv2.waitKey(1) & 0xFF

            # Enter toggles recording (some environments use 13, others 10)
            if key in (13, 10):
                if not is_logging:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    csv_path = f"{record_path}/depth_sum_positive_{ts}.csv"
                    csv_file = open(csv_path, "w", newline="")
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(["timestamp", "depth_sum_positive"])
                    csv_file.flush()
                    is_logging = True
                    log_message(f"Recording START: {csv_path}")
                else:
                    is_logging = False
                    if csv_file is not None:
                        csv_file.flush()
                        csv_file.close()
                    csv_file = None
                    csv_writer = None
                    log_message("Recording STOP")

            # q to quit
            if key == ord("q"):
                break

            # Window close handling
            if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                for _ in range(5):
                    cv2.waitKey(1)
                break

    except KeyboardInterrupt:
        log_message("Exiting...")
    finally:
        # Close CSV if still open
        if csv_file is not None:
            csv_file.flush()
            csv_file.close()

        # Release the camera and close windows.
        if cam_stream.camera is not None:
            cam_stream.camera.release()
        cv2.destroyAllWindows()
        if visualizer3D:
            visualizer3D.visualizer.destroy_window()


if __name__ == "__main__":
    import argparse
    from config import GSConfig

    parser = argparse.ArgumentParser(
        description="Run the Gelsight Mini Viewer with an optional config file."
    )
    parser.add_argument(
        "--gs-config",
        type=str,
        default=None,
        help="Path to the JSON configuration file. If not provided, default config is used.",
    )

    args = parser.parse_args()

    if args.gs_config is not None:
        log_message(f"Provided config path: {args.gs_config}")
    else:
        log_message(f"Didn't provide custom config path.")
        log_message(
            f"Using default config path './default_config.json' if such file exists."
        )
        log_message(
            f"Using default_config variable from 'config.py' if './default_config.json' is not available"
        )
        args.gs_config = "default_config_record_1.json"

    gs_config = GSConfig(args.gs_config)
    View3D(config=gs_config.config)
