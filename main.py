import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict,deque
import os
def is_inside_roi(box, roi):
    xmin, ymin, xmax, ymax = map(int, box)
    center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2
    return cv2.pointPolygonTest(roi, (center_x, center_y), False) >= 0

def draw_roi(frame,roi,fill=True):
    frame = cv2.polylines(frame, [roi], isClosed=True, color=(0, 255, 0), thickness=1)

    if fill:
        mask = np.zeros_like(frame, dtype=np.uint8)
        cv2.fillPoly(mask, [roi], (0, 255, 0))
        roi_mask = cv2.bitwise_and(mask, frame)
        frame = cv2.addWeighted(frame, 0.7, roi_mask, 0.3, 0)
    return frame

def draw_corner_box(frame, bbox, color=(255, 200, 100), thickness=3, corner_length=10):
    xmin, ymin, xmax, ymax = bbox

    cv2.line(frame, (xmin, ymin), (xmin + corner_length, ymin), color, thickness)
    cv2.line(frame, (xmin, ymin), (xmin, ymin + corner_length), color, thickness)

    cv2.line(frame, (xmax, ymin), (xmax - corner_length, ymin), color, thickness)
    cv2.line(frame, (xmax, ymin), (xmax, ymin + corner_length), color, thickness)

    cv2.line(frame, (xmin, ymax), (xmin + corner_length, ymax), color, thickness)
    cv2.line(frame, (xmin, ymax), (xmin, ymax - corner_length), color, thickness)

    cv2.line(frame, (xmax, ymax), (xmax - corner_length, ymax), color, thickness)
    cv2.line(frame, (xmax, ymax), (xmax, ymax - corner_length), color, thickness)

def draw_analyst(frame,analyst_size):
    analyst = np.ones((analyst_size[1],analyst_size[0],3),dtype=np.uint8) * 255
    cv2.rectangle(analyst, (0, 0), (analyst_size[0] - 1, analyst_size[1] - 1), (0, 0, 255), 5)
    frame[10:10 + analyst_size[1],10:analyst_size[0] + 10] = analyst
    return frame

def draw_minimap(frame, track_positions_main_player, scale_x, scale_y,court,minimap_size):

    minimap = np.ones((minimap_size[1], minimap_size[0], 3), dtype=np.uint8) * 255

    for obj_id,data in track_positions_main_player.items():
        for box_obj in data:
            xmin, ymin, xmax, ymax = box_obj
            center_x = (xmin + xmax) // 2
            # center_y = (ymin + ymax) // 2

            center_x = int(center_x * scale_x)
            # center_y = int(center_y * scale_y)
            ymax = int(ymax * scale_y)
            roi = []
            for box in court:
                x,y = box
                x = int(x * scale_x)
                y = int(y * scale_y)
                roi.append((x,y))
            roi = np.array(roi,dtype=np.int32)
            minimap = draw_roi(minimap,roi,True)


            cv2.circle(minimap, (center_x, ymax), 5, (0, 0, 255), -1)
            cv2.putText(minimap, "{}".format(obj_id), (center_x + 10, ymax), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.rectangle(minimap,(0,0),(minimap_size[0]-1,minimap_size[1]-1),(0,0,255),3)


    frame[100:100+ minimap_size[1], -minimap_size[0]-20:-20] = minimap
    return frame
if __name__ == '__main__':
    model = YOLO('data/yolov8s.pt')

    cap = cv2.VideoCapture('data/mel.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    delay = int(1000 / fps)

    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    os.makedirs("output", exist_ok=True)
    out = cv2.VideoWriter("output/tenis_match.mp4", fourcc, fps, (1000, 800))
    # cap.set(cv2.CAP_PROP_POS_FRAMES, fps * 16)
    track_positions = defaultdict(lambda: deque([], maxlen=2))
    track_positions_main_player = defaultdict(lambda: deque([], maxlen=1))
    track_total_distance = defaultdict(lambda: {"total_km": 0})
    y_offset = 35

    roi = np.array([(330, 110), (670, 110), (900, 750), (114, 750)])
    court = [(307, 200), (695, 200), (852, 595), (154, 595)]
    analyst_size = (330, 100)
    minimap_size = (200, 200)
    scale_x = minimap_size[0] / 1000
    scale_y = minimap_size[1] / 800
    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break
        frame = cv2.resize(frame, (1000, 800))
        results = model.track(frame,tracker='bytetrack.yaml',persist=True)
        frame = draw_analyst(frame,analyst_size)

        for cls,track_id,box in zip(results[0].boxes.cls,results[0].boxes.id,results[0].boxes.xyxy):
            if int(cls) == 0 and track_id is not None:  # class "person"
                track_id = int(track_id.item())
                xmin, ymin, xmax, ymax = map(int, box.tolist())
                track_positions[track_id].append((xmin, ymin,xmax, ymax))

                if len(track_positions[track_id]) > 1:
                    xmin_1, ymin_1, xmax_1, ymax_1 = track_positions[track_id][0]
                    xmin_2, ymin_2, xmax_2, ymax_2 = track_positions[track_id][1]
                    center_x1,center_y1 = (xmin_1 + xmax_1) // 2, (ymin_1 + ymax_1) // 2
                    center_x2, center_y2 = (xmin_2 + xmax_2) // 2, (ymin_2 + ymax_2) // 2


                    pixel_distance = ((center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2) ** 0.5
                    meter_distance = pixel_distance * 0.01
                    track_total_distance[track_id]["total_km"] = track_total_distance[track_id][
                                                                     "total_km"] + meter_distance

                for id_obj, data in track_total_distance.items():
                    if id_obj in track_positions and track_positions[id_obj]:

                        xmin_, ymin_, xmax_, ymax_ = track_positions[id_obj][-1]
                        center_x, center_y = (xmin_ + xmax_) // 2, (ymin_ + ymax_) // 2
                        bbox = [center_x - 1, center_y - 1, center_x + 1, center_y + 1]

                        if data["total_km"] > 100 or is_inside_roi(bbox,roi):
                            track_positions_main_player[id_obj].append((xmin_, ymin_, xmax_, ymax_))
                            cv2.putText(frame, "Player {}".format(id_obj), (xmin_, ymin_ - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            draw_corner_box(frame, map(int, [xmin_, ymin_, xmax_, ymax_]))
                            cv2.putText(frame, "Player {} moved: {:.2f} m".format(id_obj, data["total_km"]),
                                        (20, y_offset),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                            y_offset += 50

                frame = draw_minimap(frame, track_positions_main_player, scale_x, scale_y, court, minimap_size)

                y_offset = 35
        out.write(frame)
        # cv2.imshow('Tracking', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Done!")


