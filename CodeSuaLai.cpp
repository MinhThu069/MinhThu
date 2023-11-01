#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>
#include <Eigen/Dense>

#define MAX 100
using namespace std;
using namespace Eigen;


string BangDiem = "Diem_OOP.txt";

class Score{
    private:
        vector <double> points;
        double examScore;
    
    public:
        Score() = default;
        Score(const vector<double>& pts, double eScore) : points(pts), examScore(eScore) {}


    // Getter và Setter
    double getPoint(int index) const {
        if (index >= 0 && index < points.size()) {
            return points[index];
        }
        return -1; // Trả về giá trị không hợp lệ nếu chỉ số không hợp lệ
    }

    void setPoint(int index, double value) {
        if (index >= 0 && index < points.size()) {
            points[index] = value;
        }
    }

    double getExamScore() const {
        return examScore;
    }

    void setExamScore(double value) {
        examScore = value;
    }

    const vector<double>& getPoints() const {
        return points;
    }
};

class ScoreTable{   
    private:
        vector<Score>scores;
    public:
    
    vector<double> pointsAvg() {                 // tính trung bình cộng các cột điểm và chứa trong vecto
        int numColumns = scores[0].getPoints().size();  // Số lượng điểm (không tính điểm thi)
        vector<double> columnAverages(numColumns, 0);   // Khởi tạo vector chứa trung bình cho mỗi cột với giá trị 0
        for (int i = 0; i < numColumns; i++) {          // Tính tổng cho mỗi cột
            for (const auto& score : scores) {
                columnAverages[i] += score.getPoint(i);
            }
        }
        for (int i = 0; i < numColumns; i++) {          // Chia tổng cho số dòng để có trung bình cho mỗi cột
            columnAverages[i] /= scores.size();
        }
        return columnAverages;
    }

    vector<double> calculateStdDev() {       // cái này tính độ lệch chuẩn
        int numPoints = scores[0].getPoints().size(); // Lấy số lượng điểm quá trình

        vector<double> columnMeans = pointsAvg(); // Lấy trung bình cho mỗi cột điểm quá trình
        vector<double> stdDevs(numPoints, 0.0);  // Khởi tạo vector độ lệch chuẩn với giá trị ban đầu là 0

        for (int i = 0; i < numPoints; i++) {   // Tính độ lệch chuẩn cho mỗi cột điểm quá trình
            double sumSquaredDifferences = 0.0; // Tổng bình phương của sự khác biệt giữa giá trị và trung bình

            for (const auto& score : scores) {
                double difference = score.getPoint(i) - columnMeans[i];
                sumSquaredDifferences += difference * difference;
            }

            stdDevs[i] = sqrt(sumSquaredDifferences / scores.size()); // Tính độ lệch chuẩn và lưu vào vector
        }

        return stdDevs; // Trả về vector độ lệch chuẩn cho mỗi cột điểm quá trình
    }
    double examScoreAvg(){   // tính trung bình cộng điểm thi
        double temp = 0.0;
        for (const auto& score : scores) {
            temp+=score.getExamScore();
        }
        return temp/scores.size();
    }

    double examScoreStdDev() {  //tính trung bình điểm thi
        double mean = examScoreAvg(); 
        double sumSquaredDifferences = 0.0; // Tổng bình phương của sự khác biệt giữa giá trị điểm thi và trung bình điểm thi

        for (const auto& student : scores) {
            double difference = student.getExamScore() - mean;
            sumSquaredDifferences += difference * difference;
        }
        return sqrt(sumSquaredDifferences / scores.size());  // Độ lệch chuẩn của điểm thi
    }

    void addScore(const Score& s) {     // them 1 nhóm điểm 
        scores.push_back(s);
    }

    
    // xử lí giá trị ngoại lai
    void normalizeData() {    
        // Duyệt qua vector scores từ cuối lên đầu (để tránh vấn đề khi xóa phần tử trong quá trình duyệt)
        for (int i = scores.size() - 1; i >= 0; i--) {
            // Kiểm tra điểm thi của sinh viên
            if (scores[i].getExamScore() == 0.0) {
                scores.erase(scores.begin() + i); // Nếu điểm thi bằng 0, xóa toàn bộ dữ liệu điểm của sinh viên đó khỏi vector scores
            }
        }
    }

    void readFromFile(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Failed to open file: " << filename << endl;
            return;
        }
        int numScores, numColumns;
        file >> numScores >> numColumns; // Số lượng điểm

        for (int i = 0; i < numScores; i++) {
            vector<double> pts(numColumns - 1);  // -1 vì cột cuối là điểm thi
            for (int j = 0; j < numColumns - 1; j++) {
                file >> pts[j];
            }
            double eScore;
            file >> eScore;

            Score score(pts, eScore);
            addScore(score);
        }   
        file.close();
    }
    void outData()
    {
        for (const auto& score : scores) {
            for (double point : score.getPoints()) {
            cout << point << "\t"; 
        }
        cout << score.getExamScore() << endl; 
        }
    }
    
    //hàm tính các hệ số hồi quy
    VectorXd multipleLinearRegression() {
        int numScores = scores.size();  // Lấy số lượng dữ liệu
        int numColumns = scores[0].getPoints().size() + 1; // +1 để tính hệ số tự do 

        MatrixXd X(numScores, numColumns); // Khởi tạo ma trận X chưa bảng điểm quá trình từng buổi (29 x5)
        VectorXd Y(numScores);  // Khởi tạo vector Y chứa điểm thi 

        // Điền dữ liệu vào X và Y
        for (int i = 0; i < numScores; i++) {
            X(i, 0) = 1; // Điền hệ số tự do, cái này điền cột đầu cho ma trận đều là 1    ví dụ:
for (int j = 0; j < numColumns - 1; j++) {                                        //          1  9.5	7	 8	 9.5   6
                X(i, j + 1) = scores[i].getPoint(j);  // Điền các điểm quá trình              //          1  7.5	6.5	 9	 8	   10
            }                                                                                 //          1  5	    0	 5	 6	   7
            Y(i) = scores[i].getExamScore(); // Điền điểm thi vào vector Y                    //          1  4	    6	 6	 6	   7
        }                                                                                     //          1  7	    7	 9	 0	   6
        // Tính toán các hệ số hồi quy                                                        //          1  8.5	6.5	 9.5 9	  10
        VectorXd B = (X.transpose() * X).ldlt().solve(X.transpose() * Y); //theo bảng công thức

        return B;  // Trả về vector chứa các hệ số hồi quy
    }

    // phân lớp dữ liệu thành k tập hợp, huấn luyện mô hình
 VectorXd K_Fold(int k) {
    VectorXd regressionCoefficients;
    vector<Score> originalScores = scores;

    int totalDataPoints = scores.size();
    int basicFoldSize = totalDataPoints / k;
    int remainder = totalDataPoints % k;
    int correctPredictions = 0;

    for (int i = 0; i < k; i++) {
        int startIdx = i * basicFoldSize;
        int endIdx = startIdx + basicFoldSize;
        if (remainder > 0) {
            endIdx++;
            remainder--;
        }

        vector<Score> testScores(originalScores.begin() + startIdx, originalScores.begin() + endIdx);
        vector<Score> trainScores = originalScores;

        // Lấy đi tập test khỏi tập train
        trainScores.erase(trainScores.begin() + startIdx, trainScores.begin() + endIdx);

        scores = trainScores; // Sử dụng tập huấn luyện để tính hệ số hồi quy

        regressionCoefficients = multipleLinearRegression();

        for (const Score& testScore : testScores) {
            double predictedValue = regressionCoefficients(0);
            for (int j = 0; j < testScore.getPoints().size(); j++) {
                predictedValue += regressionCoefficients(j + 1) * testScore.getPoint(j);
            }
            if (round(predictedValue) == round(testScore.getExamScore())) {
                correctPredictions++;
            }
        }
    }

    double accuracy = static_cast<double>(correctPredictions) / totalDataPoints;
    cout << "Do chinh xac: " << accuracy << endl;

    return regressionCoefficients;
}







    // thực hiện hồi quy
    VectorXd predict(const VectorXd& B) {
        int totalScores = scores.size(); 
        int totalFeatures = scores[0].getPoints().size();

        MatrixXd featureMatrix(totalScores, totalFeatures + 1); // +1 vì có một cột cho hệ số tự do

        // Xây dựng ma trận đặc trưng
        for (int i = 0; i < totalScores; i++) {
// Cột đầu tiên của ma trận là 1 - để đại diện cho hệ số tự do
            featureMatrix(i, 0) = 1;

            for (int j = 0; j < totalFeatures; j++) {
                featureMatrix(i, j + 1) = scores[i].getPoint(j);
            }
        }
        // Thực hiện phép nhân ma trận để dự đoán giá trị 
        return featureMatrix * B;
    }


    // Tính toán độ chính xác
    double calculateR2(const VectorXd& y_pred) {
        double y_mean = examScoreAvg();
        double ss_tot = 0;
        double ss_res = 0;

        for (int i = 0; i < scores.size(); i++) {
            double y_actual = scores[i].getExamScore();

            ss_tot += (y_actual - y_mean) * (y_actual - y_mean);
            ss_res += (y_actual - y_pred(i)) * (y_actual - y_pred(i));
        }

        return 1 - (ss_res / ss_tot);
    }   
};  


void Run()
{
    ScoreTable table;        //khai báo bảng điểm
    table.readFromFile(BangDiem);     //đọc dữ liệu
    //table.normalizeData();
    VectorXd coefficients = table.K_Fold(6);     // tính hệ số hồi quy
    VectorXd predictions = table.predict(coefficients);
    double r2 = table.calculateR2(predictions);
    cout << "\nDo chinh xac: " << r2 << endl;
    // in ra ket qua du doan
    cout << "\nDu doan:" << endl;
    for (int i = 0; i < predictions.size(); i++) {
        cout << "Du doan cho mau " << i+1 << ": " << predictions(i) << endl;
    }
}

int main() {
    Run();
    return 0;
}