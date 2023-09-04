#include <iostream>
#include <stdio.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <set>
#include <istream>
#include <string>
#include <cmath>
#include <unistd.h>
#include <numeric>

// xtensor
#include <xtensor/xarray.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

// xsimd
#include "xsimd/xsimd.hpp"

using namespace std;
using namespace cv;
using namespace std::chrono;


vector<string> getCOCO(const string filePath) {
    ifstream txtFile (filePath);
    string class_name;
    vector<string> coco_classes;

    while(getline(txtFile, class_name)) {
        coco_classes.push_back(class_name);
    }

    return coco_classes;
}

float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

/*
 * @param predictions the array to be transpose
 * @ param dims the goal dimensions 
 * 
 */
xt::xarray<float> transpose(xt::xarray<float>& predictions) {
    
    xt::xarray<float>::shape_type shape = {predictions.shape()[0], predictions.shape()[2], predictions.shape()[3], predictions.shape()[1]};
    xt::xarray<float> new_predictions(shape);

    for (std::size_t n = 0; n < predictions.shape()[0]; n++) {
        for (std::size_t h = 0; h < predictions.shape()[2]; h++) {
            for (std::size_t w = 0; w < predictions.shape()[3]; w++) {
                for (std::size_t c = 0; c < predictions.shape()[1]; c++) {
                    new_predictions(n, h, w, c) = predictions(n, c, h, w);
                }
            }
        }
    }
    return new_predictions;
}

/*
 *@param predictions the array that stores first read npy file values
 *
 */
void find_maximum(xt::xarray<float>& predictions, int NUM_CLASS) {

    auto batch_size = predictions.shape()[0];
    predictions = transpose(predictions);

    std::cout << predictions.shape()[1] << std::endl;

    predictions.reshape({batch_size, predictions.shape()[1], predictions.shape()[2], 3, (5 + NUM_CLASS)});

    for (std::size_t j = 0; j < predictions.shape()[3]; j++) {
        float max_class_value = sigmoid(predictions(0, 0, 0, j, 5));
        int indx = 5;
        for (std::size_t i = 5; i < predictions.shape()[4]; i++) {
            float temp = sigmoid(predictions(0, 0, 0, j, i));
            if (temp > max_class_value) {
                max_class_value = temp;
                indx = i;
            }
    }
    std::cout << "object index: " << indx-5 << "object value: " << max_class_value << std::endl;
  }

}

xt::xarray<float> reshape_tensor(string filePath, int NUM_CLASS) {

    auto data = xt::load_npy<float>(filePath);
  
    auto batch_size = data.shape()[0];
    xt::xarray<float> predictions = data;

    predictions = transpose(predictions);

    predictions.reshape({batch_size, predictions.shape()[1] * predictions.shape()[2] * 3, (5 + NUM_CLASS)});

    return predictions;
}

/*
 *TO-DO: check if each box score pass the threshold = e.g. 0.99 in t99
 *@param box conf score
 *@param threshold = 0.99
 *
 *@return one-dimensional index (within the 507 boxes) array in which the box score passes the threshold
 *
 */
vector<int> box_indx(vector<float>& box_conf, float conf_thres) {
    vector<int> box_ind;
    for (std::size_t i = 0; i < box_conf.size(); i++) {
        if (box_conf[i] >= conf_thres) {
            box_ind.push_back(i);
        }
    }

    return box_ind;
}

/*
 *@param : 
 *@param: row, col, id
 * 
 * 
 */
vector<float> get_abs_detect_boxes(const vector<float>& bbox_xywh, const vector<int>& rcn , vector<float>& anchors, int coord_norm, vector<int>& size_norm) {
    vector<float> abs_xywh;
    abs_xywh.push_back((rcn[1] + bbox_xywh[0]) / coord_norm); //x
    abs_xywh.push_back((rcn[0] + bbox_xywh[1]) / coord_norm); //y

    abs_xywh.push_back(exp(bbox_xywh[2]) * anchors[0] / size_norm[0]); //width
    abs_xywh.push_back(exp(bbox_xywh[3]) * anchors[1] / size_norm[1]); //height

    return abs_xywh;
}

/*
 *TO-DO: to get the row and column of the bbox in the S*S grid image
 *
 *@param obj_index: the index of the object in ex. 507s bboxes
 *@param grid_size: the size of image splitted into S*S grids -> the S value
 * 
 *@return row, col, n
 *
 */
vector<int> getLocation(int obj_index, int grid_size) {
    vector<int> rcn;
    int row = obj_index / (grid_size * 3);
    rcn.push_back(row);
    int col = (obj_index - row * grid_size * 3) / 3;
    rcn.push_back(col);
    int n = (obj_index - row * grid_size * 3) % 3;
    rcn.push_back(n);
    return rcn;
}

Point2f bbox_minXY(const vector<float>& bbox_xywh) {
    Point2f min_xy;
    float xmin = bbox_xywh[0] - bbox_xywh[2] / 2;
    float ymin = bbox_xywh[1] - bbox_xywh[3] / 2;

    min_xy.x = xmin;
    min_xy.y = ymin;

    return min_xy;
}

Point2f bbox_maxXY(const vector<float>& bbox_xywh) {
    Point2f max_xy;
    float xmax = bbox_xywh[0] + bbox_xywh[2] / 2;
    float ymax = bbox_xywh[1] + bbox_xywh[3] / 2;
    
    max_xy.x = xmax;
    max_xy.y = ymax;

    return max_xy;
}

/*
 *TO-DO: get the iou result
 *@param bbox1: selected box
 *@param bbox2: other boxes to check overlap and calculate iou
 *
 *@return iou result
 *
 */
float IoU(const vector<float>& bbox1, const vector<float>& bbox2) {

    float width_of_overlap_area = min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]);
    float height_of_overlap_area = min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]);
    
    float overlap_area;
    if ((width_of_overlap_area < 0) || (height_of_overlap_area < 0)) {
        overlap_area = 0;
    } else {
        overlap_area = width_of_overlap_area * height_of_overlap_area;
    }

    float bbox1_area = (bbox1[3] - bbox1[1]) * (bbox1[2] - bbox1[0]);
    float bbox2_area = (bbox2[3] - bbox2[1]) * (bbox2[2] - bbox2[0]);

    float union_bbox_area = bbox1_area + bbox2_area - overlap_area;
    
    if (union_bbox_area == 0) {
        return 0;
    }

    // return the iou result
    return overlap_area / union_bbox_area;
}

/*
 *TO-DO: to sort the detected boxes by its conf_score
 *
 */
bool sortDetectedBBox(const vector<float>& bbox1, const vector<float>& bbox2) {
    return bbox1[5] > bbox2[5];
}

/*
 *TO-DO: do nms to filter out all the best bboxes
 *@param detected_boxes: the indexes of all the bboxes that pass the threshold of 0.99 -> {xmin, ymin, xmax, ymax, id, score(result)}
 *@param iou_thres: the iou threshold to check whether the iou result passes the iou threshold
 *
 *@return the selected index of each selected bbox
 *
 */
vector<vector<float>> nms(vector<vector<float>>& detected_boxes, float iou_thres) {

    std::sort(detected_boxes.begin(), detected_boxes.end(), sortDetectedBBox);

    for (std::size_t i = 0; i < detected_boxes.size(); i++) {

        if(detected_boxes[i][5] == 0) {
            continue;
        }

        for (std::size_t j = i + 1; j < detected_boxes.size(); j++) {
            if (detected_boxes[i][4] != detected_boxes[j][4]) {
                continue;
            }

            if (IoU(detected_boxes[i], detected_boxes[j]) > iou_thres) {
                detected_boxes[j][5] = 0.0;
            }
        }
    }

    // try: find_if
    vector<vector<float>> remain_obj;
    for (std::size_t i = 0; i < detected_boxes.size(); i++) {
        if (detected_boxes[i][5] > 0) {
            remain_obj.push_back(detected_boxes[i]);
        }
    }

    return remain_obj;
}

vector<vector<float>> resize_detected_box(vector<vector<float>>& detected_boxes, const vector<int>& origin_img_size) {
    for (std::size_t i = 0; i < detected_boxes.size(); i++) {
        detected_boxes[i][0] *= origin_img_size[0]; //xmin
        detected_boxes[i][2] *= origin_img_size[0]; //xmax
        detected_boxes[i][1] *= origin_img_size[1]; //ymin
        detected_boxes[i][3] *= origin_img_size[1]; //ymax
    }

    return detected_boxes;
}

vector<vector<float>> clip_detections(vector<vector<float>>& detected_boxes, const vector<int>& origin_img_size) {
    for (std::size_t i = 0; i < detected_boxes.size(); i++) {
        detected_boxes[i][0] = max(int(detected_boxes[i][0]), 0);
        detected_boxes[i][1] = max(int(detected_boxes[i][1]), 0);
        detected_boxes[i][2] = min(int(detected_boxes[i][2]), origin_img_size[1]);
        detected_boxes[i][3] = min(int(detected_boxes[i][3]), origin_img_size[0]);
    }

    return detected_boxes;
}

int main (int argc, char *argv[]) {

    if (argc != 2) {
        printf("usage: ./pp <Image_Path>\n");
        return -1;
    }

    Mat image;
    image = imread(argv[1], IMREAD_COLOR);

    if (!image.data)
    {
        printf("No image data \n");
        return -1;
    }

    const string COCO_FILE = "/workspace/data/coco.names";

    vector<string> coco_list = getCOCO(COCO_FILE);
    const int NUM_CLASS = coco_list.size();

    vector<float> drawresult;

    vector<float> confidence_threshold = {0.99, 0.6, 0.1};

    for (std::size_t th = 0; th < confidence_threshold.size(); th++) {
        vector<float> processresult;
        vector<float> reshape_time_result;
        vector<float> filter_result;
        vector<float> nms_time_result;

        for (std::size_t run = 0; run < 100; run++) {
            // here to read the three layer outputs result npy files
            vector<string> layer_outputs = {"../data/conv2d_58_Conv2D_YoloRegion.npy", "../data/conv2d_74_Conv2D_YoloRegion.npy", "../data/conv2d_66_Conv2D_YoloRegion.npy"};

            vector<vector<float>> anchors {{116.0, 90.0, 156.0, 198.0, 373.0, 326.0},
                                           {10.0, 13.0, 16.0, 30.0, 33.0, 23.0},
                                           {30.0, 61.0, 62.0, 45.0, 59.0, 119.0}};

            // the input shape of each layer
            vector<int> input_shape = {416, 416, 3};

            // not yet used
            vector<int> resized_shape = {416, 416, 3};
            vector<int> original_shape = {720, 1280, 3};

            // the iou threshold
            float iou_threshold = 0.15;

            vector<vector<float>> detected_boxes;
            auto processStart = high_resolution_clock::now();

            for (std::size_t l = 0; l < layer_outputs.size(); l++) {

                string filePath = layer_outputs[l];

                // check maximum values (not used)
                // xt::xarray<float> data = xt::load_npy<float>(filePath);
                // find_maximum(data, NUM_CLASS);
            
                auto reshapeStart = high_resolution_clock::now();
                auto predictions = reshape_tensor(filePath, NUM_CLASS);
                auto reshapeEnd = high_resolution_clock::now();

                auto reshape_Dura = duration_cast<chrono::microseconds>(reshapeEnd - reshapeStart);
                reshape_time_result.push_back(reshape_Dura.count());


                float confidence_score, class_confidence_score, result_prob;

                vector<float> result_score; 

                auto filterStart = high_resolution_clock::now();

                for (std::size_t j = 0; j < predictions.shape()[1]; j++) {
                    confidence_score = predictions(0, j, 4);

                    // mean that there will be 507 * 85
                    for (std::size_t i = 5; i < predictions.shape()[2]; i++) {
                        class_confidence_score = predictions(0, j, i);

                        result_prob = confidence_score * class_confidence_score;
                        result_score.push_back(result_prob);
                    }
                }

                vector<float> valid_score;

                for (std::size_t i = 0; i < result_score.size(); i++) {
                    if (result_score[i] >= confidence_threshold[th]) {
                        valid_score.push_back(result_score[i]);
                    }
                }

                auto obj_indx = box_indx(result_score, confidence_threshold[th]);

                vector<vector<float>> bbox_xywh;

                for (std::size_t i = 0; i < obj_indx.size(); i++) {
                    vector<float> temp_save;
                    temp_save.push_back(predictions(0, obj_indx[i] / NUM_CLASS, 0));
                    temp_save.push_back(predictions(0, obj_indx[i] / NUM_CLASS, 1));
                    temp_save.push_back(predictions(0, obj_indx[i] / NUM_CLASS, 2));
                    temp_save.push_back(predictions(0, obj_indx[i] / NUM_CLASS, 3));
                    bbox_xywh.push_back(temp_save);
                    vector<float> temp_det_bbox;
                    vector<float> temp_anchor(anchors[l].begin() + 2 * getLocation(obj_indx[i] / NUM_CLASS, (int)sqrt(predictions.shape()[1] / 3))[2], anchors[l].begin() + 2 * getLocation(obj_indx[i] / NUM_CLASS, (int)sqrt(predictions.shape()[1] / 3))[2] + 2);
                    vector<float> value = get_abs_detect_boxes(bbox_xywh[i], getLocation(obj_indx[i] / NUM_CLASS, (int)sqrt(predictions.shape()[1] / 3)), temp_anchor, (int)sqrt(predictions.shape()[1] / 3), input_shape);

                    temp_det_bbox.push_back(value[0]);
                    temp_det_bbox.push_back(value[1]);
                    temp_det_bbox.push_back(value[2]);
                    temp_det_bbox.push_back(value[3]);

                    temp_det_bbox.push_back(obj_indx[i] % NUM_CLASS);
                    vector<float> right_temp;
                    float xmin = bbox_minXY(temp_det_bbox).x;
                    right_temp.push_back(xmin);
                    float ymin = bbox_minXY(temp_det_bbox).y;
                    right_temp.push_back(ymin);
                    float xmax = bbox_maxXY(temp_det_bbox).x;
                    right_temp.push_back(xmax);
                    float ymax = bbox_maxXY(temp_det_bbox).y;
                    right_temp.push_back(ymax);
                    right_temp.push_back(temp_det_bbox[4]);
                    right_temp.push_back(valid_score[i]);

                    detected_boxes.push_back(right_temp);
                }

                auto filterEnd = high_resolution_clock::now();
                auto filter_duration = duration_cast<chrono::microseconds>(filterEnd - filterStart);
                filter_result.push_back(filter_duration.count());
            }

            auto nmsStart = high_resolution_clock::now();

            vector<vector<float>> remain_object = nms(detected_boxes, iou_threshold);

            auto nmsEnd = high_resolution_clock::now();
            auto nms_duration = duration_cast<chrono::microseconds>(nmsEnd - nmsStart);
            nms_time_result.push_back(nms_duration.count());

            std::swap(original_shape[0], original_shape[1]);
            remain_object = resize_detected_box(remain_object, original_shape);
            std::swap(original_shape[1], original_shape[0]);
            remain_object = clip_detections(remain_object, original_shape);
            

            vector<double> scores;
            vector<Point> minPoints;
            vector<Point> maxPoints;
            vector<int> ids;
            vector<string> labels;

            for (std::size_t i = 0; i < remain_object.size(); i++) {
                vector<string> each_set;
                minPoints.push_back(Point((int)remain_object[i][0], (int)remain_object[i][1]));
                maxPoints.push_back(Point((int)remain_object[i][2], (int)remain_object[i][3]));
                ids.push_back((int)remain_object[i][4]);
                labels.push_back(coco_list[remain_object[i][4]]);
                scores.push_back(remain_object[i][5]);
            }

            auto processEnd = high_resolution_clock::now();
            auto durationProcess = duration_cast<chrono::microseconds>(processEnd - processStart);
            processresult.push_back(durationProcess.count());


            auto drawStart = high_resolution_clock::now();
            
            Scalar box_color;
            int i = 0;

            while (i < scores.size()) {
                switch (ids[i]) {
                    case 0:
                        box_color = Scalar(206, 253, 188);
                        break;
                    case 2:
                        box_color = Scalar(147, 20, 255);
                        break;
                    case 5:
                        box_color = Scalar(230, 33, 145);
                        break;
                    case 7:
                        box_color = Scalar(252, 228, 77);
                        break;
                    case 9:
                        box_color = Scalar(63, 198, 247);
                        break;
                    default:
                        break;
                }

                rectangle(image, minPoints[i], maxPoints[i], box_color, 2, LINE_8);
                putText(image, labels[i].append(' ' + to_string(scores[i] * 100) + '%'), Point(minPoints[i].x, minPoints[i].y), FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2, LINE_AA);
                labels.clear();
                i++;
            }
            
            auto drawEnd = high_resolution_clock::now();
            auto durationDraw = duration_cast<chrono::microseconds>(drawEnd - drawStart);
            drawresult.push_back(durationDraw.count());

            namedWindow("Display Image", WINDOW_AUTOSIZE);
            imshow("Display Image", image);
            // waitKey(0);
            destroyAllWindows();

        }
        

        std::cout << "\n" << std::endl;
        std::cout << "------------------------------ threshold " << confidence_threshold[th] << " (in millisecond)------------------------------" << std::endl;
        std::cout << "reshape Process: " << accumulate(reshape_time_result.begin(), reshape_time_result.end(), 0) / 100 << std::endl;
        std::cout << "filter Process : " << accumulate(filter_result.begin(), filter_result.end(), 0) / 100 << std::endl;
        std::cout << "nms Process    : " << accumulate(nms_time_result.begin(), nms_time_result.end(), 0) / 100 << std::endl;
        std::cout << "process Process: " << accumulate(processresult.begin(), processresult.end(), 0) / 100 << std::endl;
        std::cout << "draw Process   : " << accumulate(drawresult.begin(), drawresult.end(), 0) / 100 << std::endl;
        std::cout << "total Process  : " << (accumulate(processresult.begin(), processresult.end(), 0) + accumulate(drawresult.begin(), drawresult.end(), 0)) / 100 << std::endl;
   
    }
    
    return 0;
}