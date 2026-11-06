# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** 3.13.18 / 2.11.0+cpu / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh**. Vì classification single-label của checkpoint này gán **một lớp cho toàn ảnh**, không tạo một box/object riêng trong JSON.
3. Class list **đi kèm taxonomy/checkpoint** (ImageNet-1K với `yolo11n-cls.pt`), không phải model tự nghĩ ra tên lớp.
4. Cần cả ba vì: `class_id` để định danh các lớp bằng mã riêng trong taxonomy được chọn,; `class_name` để giúp ta biết lớp đó tên là gì dễ đọc cho người; `taxonomy_name` để biết các lớp này thuộc hệ thống nhãn nào, tránh nhầm lẫn khi các taxonomy sử dụng cùng một class_id.
5. Guideline cần nói rõ chọn lớp theo tiêu chí nào khi ảnh nhiều chủ đề (ví dụ: chủ thể lớn nhất / chủ đề chính của cảnh / lớp xuất hiện rõ nhất), để mọi annotator chọn thống nhất **một** lớp.
6. Vì `score` chỉ là độ tin của model trên prediction; nó không được người xác nhận theo guideline, và top-1 ImageNet trên ảnh COCO có thể trông “lạ” nhưng vẫn chỉ là prediction.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=bowl ; `score`=0.719062 ; `bbox_xyxy`=[32.65, 342.12, 100.16, 384.93] ; `bbox_width`=67.51 ; `bbox_height`=42.81
2. Lớp `bowl`, score 0.719062; box từ góc trên-trái khoảng pixel (32.65, 342.12) đến góc dưới-phải (100.16, 384.93); rộng khoảng 67.51 px, cao khoảng 42.81 px.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
- threshold 0.20 có 17 prediction, threshold 0.6 có 6 prediction
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
- Kiểm tra thêm các prediction có confidence thấp.
- Xác định đối tượng có được nhận diện chính xác hay không.
- Loại bỏ những box sai hoặc nhận diện nhầm.
- Điều chỉnh bounding box nếu box không bao phủ đúng đối tượng.
- Kiểm tra các đối tượng bị bỏ sót hoặc bị che khuất.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
- Bounding box phải bao quanh toàn bộ phần đối tượng nhìn thấy được trong ảnh, sát với đường viền của đối tượng nhất có thể. 
- Không bao gồm các vùng nền không thuộc đối tượng. 
- Nếu đối tượng bị che khuất, chỉ vẽ box bao quanh phần nhìn thấy được, không tự suy đoán phần bị che.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
 - Trên ảnh `kitchen`, có một chiếc bát (bowl) ở góc dưới bên trái bị cắt một phần bởi mép ảnh.
 - Tôi sẽ chọn guideline nếu đã có quy tắc rõ ràng, ví dụ: nếu còn ít nhất 50% đối tượng nằm trong khung hình thì vẫn tiến hành gán nhãn và bounding box chỉ bao quanh phần nhìn thấy được. Ngược lại, nếu đối tượng bị che khuất quá nhiều khiến không thể xác định chính xác đó là vật gì hoặc không chắc chắn về class, tôi sẽ escalation lên Lab Coach để được hướng dẫn, thay vì tự suy đoán hoặc gán nhãn không chính xác.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0,  71.0], [443.0, 72.0], [442.0, 72.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
- So với bounding box chỉ bao quanh vật thể bằng hình chữ nhật, polygon có thể mô tả hình dạng vật thể sát hơn nhờ nhiều điểm tọa độ tạo thành đường viền
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
- `instance_id` dùng để phân biệt từng object trong output lab (vd. hai `bowl` khác nhau là `kitchen-001` và `kitchen-002`). Nó không phải class_id vì class_id dùng để xác định vật thể thuộc lớp nào, cũng không phải tracking_id vì tracking_id dùng để theo dõi cùng một vật thể qua nhiều frame.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
- Khi vẽ mask, người gán nhãn cần bám sát đường viền của vật thể, không bao gồm vùng nền xung quanh để đảm bảo xác định chính xác hình dạng và phạm vi của vật thể
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
- Chiếc nồi màu bạc trong ảnh `kitchen` bị bàn che khuất khoảng 50%. Em sẽ kiểm tra guideline xem đã có quy tắc vẽ mask cho vật thể bị che khuất chưa. Nếu đã có thì làm theo, còn nếu chưa có hoặc hướng dẫn chưa rõ thì sẽ hỏi Lab Coach (escalation) để xác định cách vẽ mask phù hợp, tránh tự ý ước lượng phần bị che khuất
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
|---|---|---|---|---|
| Phân loại ảnh | Mỗi ảnh chỉ có một nhãn đại diện cho nội dung chính của toàn bộ hình ảnh. | Kết quả dự đoán Top-1 của mô hình có thể không chính xác hoặc không phù hợp với nội dung thực tế của ảnh. | Quan sát toàn bộ ảnh, tham khảo hướng dẫn và chọn một nhãn phù hợp nhất. | Kiểm tra nhãn có đúng với nội dung ảnh và bộ phân loại quy định hay không. Đảm bảo không nhầm điểm số dự đoán với nhãn thực tế. |
| Phát hiện vật thể | Mỗi vật thể có một nhãn phân loại và một khung hình chữ nhật (Bounding Box) xác định vị trí của nó. | Khi giảm ngưỡng nhận diện (Confidence Threshold), mô hình có thể tạo thêm các khung bao không chính xác. Vật thể nằm sát mép ảnh hoặc bị cắt một phần cũng khó xác định. | Xác định các vật thể cần gán nhãn, chọn đúng loại và vẽ Bounding Box bao quanh từng vật thể theo hướng dẫn. | Kiểm tra nhãn và vị trí Bounding Box, đảm bảo không bỏ sót, không gán nhãn thừa hoặc nhầm vật thể. Chú ý các trường hợp bị che khuất hoặc nằm ở mép ảnh. |
| Instance segmentation | Mỗi vật thể được gán một nhãn phân loại và một vùng mặt nạ (Mask/Polygon) thể hiện chính xác hình dạng của nó. | Khó xác định ranh giới giữa các vật thể khi chúng nằm sát nhau, chồng lấn hoặc có đường viền không rõ ràng. | Xác định từng vật thể riêng biệt, chọn đúng nhãn và vẽ Mask theo đường viền thực tế của từng vật thể. | Kiểm tra Mask có bao phủ đúng vật thể không, có lấn sang vật thể khác không và có bỏ sót instance nào cần gán nhãn hay không. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
- Em chỉ sử dụng dữ liệu được phép trong phạm vi bài lab, không tự ý chia sẻ dữ liệu cá nhân hoặc dữ liệu nhạy cảm như cá nhân, khuôn mặt, biển số hay dữ liệu nội bộ lên Colab/GitHub.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
- Em sẽ dừng xử lý và báo ngay cho Lab Coach/ / GV phụ trách lab để được hướng dẫn, không tự ý sử dụng dữ liệu ngoài phạm vi

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
