#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

typedef struct FaceInfo {
	cv::Rect rect;
	float score;
} FaceInfo;

float sigmod(float x)
{
	return 1.0 / (1.0 + exp(-x));
}

std::vector<FaceInfo> yolo_face_detect(cv::dnn::Net &detect_net, cv::Mat img)
{
	std::vector<FaceInfo> result;
	if (detect_net.empty() || !img.data) {
		return result;
	}
	
	int INPUT_SIZE = 56;
	int iw = img.cols;
	int ih = img.rows;
	cv::Mat input_blob = cv::dnn::blobFromImage(img, 1 / 255.0, cv::Size(INPUT_SIZE, INPUT_SIZE), cv::Scalar(0, 0, 0), true);
	detect_net.setInput(input_blob, "data");
	const std::vector<cv::String>  targets_node{ "layer33-conv" };
	std::vector<cv::Mat> targets_blobs;
	detect_net.forward(targets_blobs, targets_node);

	float data[1024];
	float *p = (float*)targets_blobs[0].data;
	int idx = 0;
	for (int i = 0; i < 7; i++) {
		for (int j = 0; j < 7; j++) {
			for (int k = 0; k < 18; k++) {
				//对标out["layer33-conv"].transpose(0, 3, 2, 1)[0]
				data[idx++] = p[i + k * 7 * 7 + 7 * j];
			}
		}
	}

	int BIAS_W[] = { 7, 12, 22 };
	int BIAS_H[] = { 12, 19, 29 };
	idx = 0;
	std::vector <cv::Rect>bboxes;
	std::vector<float> scores;
	for (int i = 0; i < 7; i++) {
		for (int j = 0; j < 7; j++) {
			for (int k = 0; k < 3; k++) {
				float *p = &data[idx++ * 6];
				float score = sigmod(p[4]);
				if (score > 0.75) {
					float x = ((sigmod(p[0]) + i) / 7.0)*iw;
					float y = ((sigmod(p[1]) + j) / 7.0)*ih;
					float w = (((BIAS_W[k]) * exp(p[2])) / INPUT_SIZE)*iw;
					float h = (((BIAS_H[k]) * exp(p[3])) / INPUT_SIZE)*ih;
					float x1 = int(x - w * 0.5);
					float x2 = int(x + w * 0.5);
					float y1 = int(y - h * 0.5);
					float y2 = int(y + h * 0.5);
					//printf("%f %f %f %f %f\n", x1, y1, x2, y2, score);
					bboxes.push_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));
					scores.push_back(score);
				}
			}
		}
	}

	std::vector<int> indices;
	cv::dnn::NMSBoxes(bboxes, scores, 0.75, 0.35, indices);

	for (int i : indices) {
		FaceInfo info;
		info.rect = bboxes[i];
		info.score = scores[i];
	}
	return result;
}

float* forward_landmark(cv::dnn::Net &landmark_net, cv::Mat img, cv::Rect rect)
{
	if (landmark_net.empty() || !img.data) {
		return NULL;
	}

	if (rect.x < 0 || rect.y < 0 || 
		rect.x + rect.width > img.cols ||
		rect.y + rect.height > img.rows) 
	{
		return NULL;
	}

	int INPUT_SIZE = 112;
	int iw = rect.width;
	int ih = rect.height;
	float sw = float(iw) / float(INPUT_SIZE);
	float sh = float(ih) / float(INPUT_SIZE);

	cv::Mat roi_img = img(rect);
	cv::Mat resize_mat;
	cv::resize(roi_img, resize_mat, cv::Size(INPUT_SIZE, INPUT_SIZE));
	cv::Mat input_blob = cv::dnn::blobFromImage(resize_mat);
	resize_mat.convertTo(resize_mat, CV_32FC1);

	//new_img = (resize_mat - 127.5) / (127.5)
	cv::Mat new_img = (resize_mat - 127.5) / (127.5);
	float *p = (float*)input_blob.data;
	float *d = (float*)new_img.data;
	int idx = 0;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 112 * 112; j++) {
			//对标input_shape = new_img.transpose(2,0,1)
			p[idx++] = d[3 * j + i];
		}
	}

	landmark_net.setInput(input_blob, "data");
	const std::vector<cv::String>  targets_node{ "bn6_3_scale" };
	std::vector<cv::Mat> targets_blobs;
	landmark_net.forward(targets_blobs, targets_node);

	float *blob = (float *)targets_blobs[0].data;
	float *points = (float *)malloc(targets_blobs[0].total() * sizeof(float));
	for (int i = 0; i < targets_blobs[0].total(); i += 2) {
		points[i] = ((blob[i] * INPUT_SIZE) * sw) + rect.x;
		points[i + 1] = ((blob[i + 1] * INPUT_SIZE) * sh) + rect.y;
	}
	return points;
}

int main(int argc, char *argv[])
{
	cv::dnn::Net detect_net = cv::dnn::readNetFromCaffe("model/yoloface-50k.prototxt", "model/yoloface-50k.caffemodel");
	cv::dnn::Net landmark_net = cv::dnn::readNetFromCaffe("model/landmark106.prototxt", "model/landmark106.caffemodel");

	cv::Mat img = cv::imread("test.jpg");
	std::vector<FaceInfo> result = yolo_face_detect(detect_net, img);
	for (FaceInfo info : result) {
		cv::Rect r = info.rect;
		if (r.x <= 0 || r.y < 0) {
			continue;
		}
		printf("face[%d %d %d %d](%f)\n", r.x, r.y, r.width, r.height, info.score);
		float *points = forward_landmark(landmark_net, img, r);
		for (int i = 0; i < 106; i++) {
			cv::circle(img, cv::Point(points[i * 2], points[i * 2 + 1]), 2, cv::Scalar(0, 255, 255), 3);
		}
		free(points);
		rectangle(img, r, cv::Scalar(255, 255, 0), 2);
	}
	cv::imshow("test_img", img);
	cv::waitKey(0);
	return 0;
}