# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV: 2A202604118**
- **Họ và tên: Đặng Nguyễn Nhật Minh**
- **SEED**: 523

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | x |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | x |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | x |

## 2. Kết quả chính

Phương pháp,F1 (test),F1 (SUB 40),p95 ms/tin,USD / 1k tin
1. BoW + LogReg (tự code),0.927,0.947,2.229,0.0
2. BoW + MLP (deep learning),0.963,0.974,0.361,0.0
3a. Encoder (MiniLM) + kNN,0.893,0.947,72.046,0.0
3b. Encoder (MiniLM) + LogReg,0.969,0.974,34.169,0.0
3c. Encoder (MiniLM) + MLP head,0.964,0.974,27.504,0.0
## 3. Dự đoán và thực tế

Dự đoán ml_lr (Learning Rate):

Dự đoán: Với lr=50, loss có thể dao động mạnh hoặc tăng vọt và không hội tụ.

Thực tế: Loss tăng vọt lên khoảng 2.7 trong những epoch đầu tiên nhưng lại giảm cực nhanh xuống gần 0 (0.007).

Giải thích: Learning rate quá lớn khiến bước cập nhật trọng số văng xa lúc đầu, nhưng do hàm mất mát (loss function) của bài toán đơn giản, mô hình vô tình vượt qua các vùng dốc và nhanh chóng rơi vào điểm tối ưu để hội tụ ổn định vào cuối chu trình huấn luyện.

Dự đoán dl_curve (DL so với ML):

Dự đoán: MLP chỉ nhỉnh hơn một chút so với LogReg, khác biệt không lớn vì cả hai đều dùng Bag-of-Words.

Thực tế: MLP (F1=0.963) tối ưu tốt hơn hẳn LogReg (F1=0.927). Đặc biệt trên biểu đồ Pareto, MLP lấn át hoàn toàn bằng việc giảm tỷ lệ lỗi (1-F1) rất tốt nhưng độ trễ cực kỳ thấp (0.36 ms).

Giải thích: Sự hiện diện của Hidden Layer và hàm phi tuyến ReLU đã giúp mô hình bắt được những tổ hợp đặc trưng phức tạp (non-linear relationships) thay vì chỉ cộng dồn trọng số các từ đơn lẻ một cách cơ học.

Dự đoán hard_eval (50 case khó):

Dự đoán: BoW sẽ thất bại nặng nề và chỉ đạt khoảng 35% accuracy.

Thực tế: BoW đạt độ chính xác 54%, cao hơn dự tính.

Giải thích: Một số case bị gài bẫy (như nhóm "Nói về spam" hay "Tưởng spam nhưng không phải") tình cờ có tổng điểm trọng số của các từ "ham" áp đảo từ khóa "spam", khiến mô hình BoW vẫn cộng trừ ra điểm z < 0.5 và đưa ra phán đoán đúng, dù nó không thực sự hiểu ngữ cảnh.
## 4. Phân tích 50 case khó

Model tụt nhiều nhất: 2. BoW + MLP (tụt sâu từ mức F1=0.974 trên tập SUB xuống độ chính xác chỉ còn 54% trên bộ 50 case, F1 còn 0.378). Nhóm case bị thất bại nặng nhất là Né từ khóa (Obfuscation) (đúng 0%) và Spam giọng tự nhiên (Natural Spam) (đúng 10%).

Case 1: #29 [Né từ khoá]

Nội dung: F R E E  e n t r y  into our weekly draw, t x t  WIN  to 80086

Cơ chế sai: Mô hình BoW nhìn văn bản dưới dạng các token rời rạc. Việc kẻ gian chèn khoảng trắng làm vỡ cấu trúc từ ("F", "R", "E", "E") khiến các từ khóa này trở thành những token OOV (out of vocabulary) đối với từ điển 2000 từ ban đầu. Mô hình hoàn toàn "mù" trước từ khóa cốt lõi và bỏ sót tin spam này.

Case 2: #12 [Spam giọng tự nhiên]

Nội dung: hi there, sorry to bother you, my friend gave me your number, I'm new in town and looking for someone to chat with, check my profile at meetlocal.club

Cơ chế sai: Nội dung được soạn thảo như một tin nhắn làm quen, giao tiếp đời thường. Tin nhắn không chứa bất kỳ từ khóa spam nhạy cảm nào, nên cả ML, DL, lẫn Encoder đều bị đánh lừa. Vector ngữ nghĩa của tin này nằm lẫn hoàn toàn vào không gian của các tin nhắn giao tiếp hợp lệ (ham), do mô hình không đủ khả năng suy luận ý đồ thao túng đằng sau đường link lạ.
## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật

Với yêu cầu SLA khắt khe (p95 < 50 ms/tin) để phục vụ quy mô 10 triệu SMS/ngày, phương án tối ưu được chọn là 2. BoW + MLP (deep learning).

Dẫn chứng số liệu: Đây là điểm nằm ở góc dưới-trái (tốt nhất) trên đường Pareto. Mô hình sở hữu tốc độ nội suy cả lô lên tới >35,000 tin/s, độ trễ p95 đạt mức cực thấp 0.361 ms/tin, nhưng mang lại chất lượng phân loại ngang ngửa các mô hình Encoder (F1=0.963). Các mô hình dùng Encoder tuy hiểu ngữ cảnh nhưng có độ trễ p95 từ 27-34 ms, quá sát ngưỡng rủi ro 50ms và đòi hỏi tài nguyên tính toán (GPU) rất tốn kém khi nhân với quy mô 10 triệu tin.

Đánh đổi: Lựa chọn này bắt buộc đánh đổi khả năng chống chịu trước các nhóm spam ngụy trang (Spam giọng tự nhiên, Thao túng, Né từ khóa).

Khắc phục: Để khắc phục, hệ thống cần được thiết lập một pipeline tiền xử lý (Text Normalization/Heuristics) chạy độc lập bằng RegEx trước khi đưa qua MLP. Bước này sẽ xử lý việc gộp từ bị cắt xén hoặc xóa các ký tự đặc biệt thừa, giúp BoW nhận diện đúng từ vựng mà không làm ảnh hưởng tới giới hạn thời gian 50ms.
## 7. AI log và điều đã học

AI Log:

Hỏi: Yêu cầu AI giải thích chi tiết ý nghĩa của việc "Broadcast division" (M_unit = M / safe_norms[:, None]) trong bước tính khoảng cách Cosine Similarity mà không dùng tới vòng lặp for (Phần 3.2).

Nhận & Sửa: AI giải thích rõ cách ma trận thao tác với mảng chiều phụ [:, None]. Không cần sửa logic, chỉ thêm hiểu biết về cách Numpy xử lý đại số tuyến tính cực kỳ hiệu quả để tối ưu throughput.

Điều đã học:

Sự khác biệt giữa Data chuẩn và Adversarial Data: Điểm số 99% trên tập test độc lập (i.i.d) có thể cực kỳ "lạc quan" và hoàn toàn sụp đổ trước những bẫy đơn giản ngoài đời thực như cố tình gõ sai chính tả hay chèn ký tự lạ.

Đánh đổi theo đường Pareto: Không phải cứ mô hình phức tạp hơn (Embedding/Transformer) là ưu việt hơn. Khi triển khai hệ thống lớn đòi hỏi tốc độ siêu tốc và tiết kiệm tài nguyên, một mô hình nông nhưng được thiết kế tính năng kỹ càng (MLP) hoàn toàn có thể "lấn át" mô hình sâu về mặt ứng dụng thực tiễn.
