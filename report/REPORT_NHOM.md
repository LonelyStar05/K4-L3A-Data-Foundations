# Báo Cáo Nhóm — Lab 7: Embedding & Vector Store

**Nhóm:** Nhóm L3A — VinUni Tuition, Scholarship & Financial Aid Retrieval  
**Thành viên:**
1. **Nguyễn Tú Tài** — MSSV: `2A202602455` — **R1: Crawling & Baseline Analysis**
2. **Trần Đại Nhân** — MSSV: `2A202602642` — **R2: Benchmark Queries & Hybrid Pipeline**
3. **Nguyễn Ngọc Bảo** — MSSV: `2A202602951` — **R3: Heading Chunker & Structure-Aware Analysis**

**Ngày:** 19/09/2026

> **Nộp 1 bản / nhóm.** Phần cá nhân (hướng tiếp cận, kết quả riêng, dự đoán…) mỗi thành viên nộp riêng trong `REPORT_CANHAN.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần nhóm: 40** = Lựa chọn tài liệu (10) + Thiết kế chiến lược (15) + Chất lượng truy xuất (10) + Thuyết trình (5).

---

## 1. Lựa chọn tài liệu (Document Set Quality) — Nhóm (10 điểm)

### Chủ đề (Domain) & Lý Do Chọn

**Chủ đề:** Dịch vụ và quy định đại học — chuyên sâu về **học phí, học bổng, hỗ trợ tài chính và khoản vay sinh viên tại VinUniversity**.

**Tại sao nhóm chọn chủ đề này?**
> Đây là nhóm thông tin có nhu cầu tra cứu cao nhưng dễ nhầm do cùng lúc tồn tại nhiều chương trình, niên khóa, đối tượng và mốc thời gian. Corpus cũng có nhiều dạng cấu trúc như bảng biểu phí, FAQ, quy trình và văn bản quy định, phù hợp để đánh giá ảnh hưởng của chunking, semantic retrieval và metadata filter.

### Danh sách tài liệu (Data Inventory)

Số ký tự dưới đây chỉ tính phần nội dung Markdown, không tính YAML frontmatter.

| # | Tên tài liệu | Nguồn (Source URL) | Ngày lấy / Phiên bản | Số ký tự | Metadata đã gán |
|---|--------------|--------------------|----------------------|-----------|-----------------|
| 1 | `duy-tri-hoc-bong-ho-tro-tai-chinh.md` — Quy định duy trì Học bổng đầu vào và Hỗ trợ tài chính | [VinUni Policy](https://policy.vinuni.edu.vn/wp-content/uploads/2025/09/GDL-SAM-004-V2.1_Tieu-chi-duy-tri-Hoc-bong-dau-vao-va-Ho-tro-tai-chinh_4.9.2025.pdf) | 2026-09-19 / GDL-SAM-004-V2.1 (04/09/2025) | 4,137 | `audience: student`, `department: student-affairs`, `category: scholarship` |
| 2 | `faq-hoc-phi-hoc-bong.md` — FAQ Học phí, Học bổng và Hỗ trợ tài chính | [VinUni Admissions](https://admissions.vinuni.edu.vn/vi/dai-hoc/cau-hoi-thuong-gap/hoc-phi-hoc-bong-va-ho-tro-tai-chinh/) | 2026-09-19 / `not-stated` | 5,823 | `audience: all`, `department: admissions`, `category: faq` |
| 3 | `ho-tro-tai-chinh-sinh-vien-dang-hoc.md` — Hỗ trợ tài chính dành cho sinh viên đang học | [VinUni Admissions](https://admissions.vinuni.edu.vn/vi/hoc-bong-va-ho-tro-tai-chinh/cu-nhan/ho-tro-tai-chinh/) | 2026-09-19 / `not-stated` | 1,695 | `audience: student`, `department: admissions`, `category: financial-aid` |
| 4 | `ho-tro-tai-chinh-tan-sinh-vien.md` — Hỗ trợ tài chính dành cho tân sinh viên | [VinUni Admissions](https://admissions.vinuni.edu.vn/vi/hoc-bong-va-ho-tro-tai-chinh/cu-nhan/ho-tro-tai-chinh/) | 2026-09-19 / `not-stated` | 1,309 | `audience: all`, `department: admissions`, `category: financial-aid` |
| 5 | `hoc-bong-cu-nhan.md` — Học bổng chương trình Đại học | [VinUni Admissions](https://admissions.vinuni.edu.vn/vi/hoc-bong-va-ho-tro-tai-chinh/cu-nhan/hoc-bong/) | 2026-09-19 / `not-stated` | 3,530 | `audience: all`, `department: admissions`, `category: scholarship` |
| 6 | `hoc-phi-cu-nhan.md` — Học phí chương trình Cử nhân năm học 2026-2027 | [VinUni Admissions](https://admissions.vinuni.edu.vn/vi/hoc-phi/cu-nhan/) | 2026-09-19 / 2026-2027 | 3,161 | `audience: all`, `department: admissions`, `category: tuition` |
| 7 | `hoc-phi-sau-dai-hoc.md` — Học phí chương trình Sau đại học năm học 2024-2025 | [VinUni Admissions](https://admissions.vinuni.edu.vn/vi/hoc-phi/sau-dai-hoc/) | 2026-09-19 / 2024-2025 | 1,251 | `audience: all`, `department: admissions`, `category: tuition` |
| 8 | `huong-dan-de-nghi-ho-tro-tai-chinh.md` — Guidelines for Student Financial Aid Support Request | [VinUni Policy](https://policy.vinuni.edu.vn/all-policies/guidelines-for-student-financial-support-request/) | 2026-09-19 / GDL-FAO-001-V2.0 (22/04/2025) | 6,419 | `audience: student`, `department: financial-aid-office`, `category: financial-aid` |
| 9 | `khoan-vay-sinh-vien.md` — Chương trình Vay vốn Sinh viên | [VinUni Admissions](https://admissions.vinuni.edu.vn/vi/hoc-bong-va-ho-tro-tai-chinh/cu-nhan/khoan-vay-sinh-vien/) | 2026-09-19 / `not-stated` | 1,255 | `audience: all`, `department: admissions`, `category: student-loan` |
| 10 | `quy-dinh-tai-chinh-bieu-phi.md` — Quy định Tài chính và Biểu phí năm học 2026-2027 | [VinUni Policy](https://policy.vinuni.edu.vn/wp-content/uploads/2026/08/VU_TS03.VN_Quy-dinh-tai-chinh-va-Bieu-phi_AY26-27_22.7.2026_Student.pdf) | 2026-09-19 / VU_TS03.VN (22/07/2026) | 33,545 | `audience: student`, `department: finance`, `category: regulation` |

**Danh sách kiểm tra quản trị dữ liệu (Data governance checklist):**
- [x] Corpus chỉ chứa trang/PDF công khai của VinUniversity; không có dữ liệu cá nhân, thông tin đăng nhập hoặc tài liệu nội bộ.
- [x] Mỗi tài liệu có `doc_id`, `title`, `source_url`, `retrieved_at`, `document_version` và `audience` trong metadata.
- [x] `data/hoc-phi-vinuni/sources.csv` khớp 1–1 với 10 file Markdown và ghi `license_or_permission: public-source`.
- [x] Trường `audience` có hai giá trị (`student`: 4 tài liệu, `all`: 6 tài liệu), đủ để kiểm thử metadata filter.

Kết quả kiểm tra bằng `python scripts/build_corpus.py validate`: **10/10 file hợp lệ**, `sources.csv: 1-1 match`, không có lỗi metadata.

### Cấu trúc Metadata (Metadata Schema)

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho truy xuất (retrieval)? |
|----------------|------|---------------|-------------------------------|
| `doc_id` | `str` | `quy-dinh-tai-chinh-bieu-phi` | Định danh ổn định cho tài liệu gốc, truy vết chunk và xóa toàn bộ chunk của một tài liệu. |
| `title` | `str` | `Học phí chương trình Cử nhân năm học 2026-2027` | Cung cấp ngữ cảnh ngắn gọn và hỗ trợ hiển thị nguồn cho câu trả lời. |
| `source_url` | `str` | `https://policy.vinuni.edu.vn/...` | Cho phép kiểm chứng câu trả lời trực tiếp từ nguồn chính thức. |
| `retrieved_at` | `date/string` | `2026-09-19` | Theo dõi thời điểm thu thập để biết dữ liệu có cần cập nhật hay không. |
| `document_version` | `str` | `VU_TS03.VN (22/07/2026)` | Tránh trộn quy định hoặc biểu phí thuộc các niên khóa khác nhau. |
| `audience` | `enum/string` | `student`, `all` | Lọc đúng đối tượng áp dụng; đặc biệt quan trọng với câu hỏi hỗ trợ tài chính có nội dung gần giống nhau. |
| `department` | `str` | `finance`, `admissions` | Thu hẹp kết quả theo đơn vị chịu trách nhiệm khi corpus được mở rộng. |
| `category` | `str` | `tuition`, `scholarship`, `financial-aid` | Phân loại nghiệp vụ để lọc đúng nhóm chính sách. |
| `program_level` | `str` | `undergraduate`, `postgraduate` | Tránh nhầm mức học phí giữa chương trình Cử nhân và Sau đại học. |
| `language` | `str` | `vi`, `en` | Hỗ trợ truy xuất song ngữ và lựa chọn pipeline phù hợp với ngôn ngữ nguồn. |

---

## 2. Thiết kế chiến lược (Strategy Design) — Nhóm (15 điểm)

> Mỗi thành viên thử **một chiến lược khác nhau** trên cùng bộ tài liệu; nhóm tổng hợp và so sánh ở đây.

### Phân tích đường cơ sở (Baseline Analysis)

R1 chạy `ChunkingStrategyComparator().compare(text, chunk_size=500)` trên phần thân đã bỏ YAML frontmatter của 3 tài liệu đại diện. Cấu hình tương ứng là `FixedSizeChunker(500, overlap=50)`, `SentenceChunker(max_sentences_per_chunk=3)` và `RecursiveChunker(chunk_size=500)`.

| Tài liệu | Chiến lược (Strategy) | Số lượng Chunk | Độ dài trung bình | Giữ được ngữ cảnh không? |
|-----------|----------|-------------|------------|-------------------|
| `hoc-phi-cu-nhan.md` (3,162 ký tự) | FixedSizeChunker (`fixed_size`) | 7 | 494.6 | Trung bình: kích thước đều nhưng có thể cắt ngang dòng bảng và cụm số tiền. |
| `hoc-phi-cu-nhan.md` (3,162 ký tự) | SentenceChunker (`by_sentences`) | 5 | 630.6 | Khá: giữ câu hoàn chỉnh, nhưng bảng Markdown ít dấu kết câu làm chunk dài tới 987 ký tự. |
| `hoc-phi-cu-nhan.md` (3,162 ký tự) | RecursiveChunker (`recursive`) | 9 | 349.8 | Khá: ưu tiên ranh giới đoạn/dòng, nhưng vẫn sinh mảnh ngắn 68 ký tự ở vùng bảng. |
| `huong-dan-de-nghi-ho-tro-tai-chinh.md` (6,420 ký tự) | FixedSizeChunker (`fixed_size`) | 15 | 474.7 | Trung bình: overlap giữ một phần ngữ cảnh nhưng có thể tách quy trình khỏi mốc thời gian. |
| `huong-dan-de-nghi-ho-tro-tai-chinh.md` (6,420 ký tự) | SentenceChunker (`by_sentences`) | 18 | 354.4 | Khá: tốt với văn xuôi, nhưng heading/bảng tạo mảnh 11 ký tự và chunk dài tới 952 ký tự. |
| `huong-dan-de-nghi-ho-tro-tai-chinh.md` (6,420 ký tự) | RecursiveChunker (`recursive`) | 16 | 399.3 | Tốt nhất trong 3 baseline: các chunk khá cân bằng (279–486 ký tự) và bám ranh giới đoạn. |
| `quy-dinh-tai-chinh-bieu-phi.md` (33,546 ký tự) | FixedSizeChunker (`fixed_size`) | 75 | 496.6 | Trung bình: ổn định về độ dài nhưng dễ chia đôi điều khoản, danh sách và hàng bảng. |
| `quy-dinh-tai-chinh-bieu-phi.md` (33,546 ký tự) | SentenceChunker (`by_sentences`) | 69 | 484.2 | Khá với đoạn văn; yếu với bảng/quy định dài, chunk lớn nhất đạt 1,772 ký tự. |
| `quy-dinh-tai-chinh-bieu-phi.md` (33,546 ký tự) | RecursiveChunker (`recursive`) | 107 | 311.6 | Khá: tôn trọng nhiều ranh giới tự nhiên nhưng phân mảnh quá nhỏ ở một số heading (min 8 ký tự). |

**Kết luận baseline của R1:** `FixedSize` là mốc so sánh nhanh và ổn định, nhưng không hiểu cấu trúc tài liệu. `Recursive` giữ ngữ cảnh tốt nhất ở tài liệu quy trình cỡ vừa; với văn bản tài chính dài chứa nhiều bảng, cần chunker theo heading/section của R3 để tránh cả cắt ngang lẫn các mảnh quá nhỏ.

### Chiến lược của từng thành viên

> Mỗi thành viên điền một khối dưới đây (copy thêm nếu nhóm có nhiều hơn 3 người).

**Thành viên 1 — Nguyễn Tú Tài (MSSV: 2A202602455, R1)**
- **Loại chiến lược:** `FixedSizeChunker(chunk_size=500, overlap=50)` — dense retrieval baseline.
- **Mô tả & lý do chọn cho chủ đề này:** R1 dùng chiến lược cố định làm đường cơ sở vì dễ tái lập, tốc độ nhanh và cô lập được ảnh hưởng của chunking khi so với các chiến lược nâng cao. Trên toàn bộ corpus, chiến lược này tạo 141 chunk, độ dài trung bình 487 ký tự, tìm thấy tài liệu liên quan trong top-3 ở 5/5 câu và đạt **7/10** theo rubric; hạn chế chính là chỉ 2/5 câu được agent trả lời hoàn toàn đúng do các bảng/điều khoản có thể bị cắt ngang.
- **Code snippet:**
```python
chunker = FixedSizeChunker(chunk_size=500, overlap=50)
chunks = chunker.chunk(text_without_frontmatter)
```

**Thành viên 2 — Trần Đại Nhân (MSSV: 2A202602642, R2)**
- **Vai trò:** Benchmark Queries & Hybrid Pipeline.
- **Phần chiến lược/kết quả:** R2 cập nhật sau khi chốt 5 benchmark query và kết quả hybrid pipeline.

**Thành viên 3 — Nguyễn Ngọc Bảo (MSSV: 2A202602951, R3)**
- **Vai trò:** Heading Chunker & Structure-Aware Analysis.
- **Phần chiến lược/kết quả:** R3 cập nhật mô tả chunker theo heading và phân tích structure-aware.

### So Sánh Giữa Các Thành Viên

| Thành viên | Chiến lược (Strategy) | Điểm truy xuất (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Nguyễn Tú Tài | FixedSize (500/50) | **7/10** | Nhanh, đơn giản, tái lập tốt; `relevant@3 = 5/5`. | Cắt theo ký tự nên dễ tách bảng/điều khoản; agent chỉ trả lời đúng hoàn toàn 2/5 câu. |
| Trần Đại Nhân | R2 cập nhật | R2 cập nhật | Hybrid retrieval kết hợp nhiều tín hiệu. | Cần báo cáo ablation và chi phí/độ phức tạp. |
| Nguyễn Ngọc Bảo | R3 cập nhật | R3 cập nhật | Tận dụng heading và cấu trúc văn bản. | Cần xử lý section quá dài và section quá ngắn. |

**Chiến lược nào tốt nhất cho chủ đề này? Tại sao?**
> *Viết 2-3 câu — đây là phần được đánh giá cao nhất (khả năng suy nghĩ & giải thích):*

---

## 3. Câu hỏi đánh giá & Chất lượng truy xuất (Retrieval Quality) — Nhóm (10 điểm)

### Câu hỏi đánh giá & Câu trả lời chuẩn (nhóm thống nhất)

> **Đúng 5 câu hỏi**, đa dạng, có thể kiểm chứng; **ít nhất 1 câu** cần lọc metadata mới trả lời tốt. Đây là bộ câu hỏi chung cho mọi thành viên chạy.

| # | Câu hỏi (Query) | Câu trả lời chuẩn (Gold Answer) | Chunk nào chứa thông tin? |
|---|-------|-------------------------------|--------------------------|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |

### Tổng hợp chất lượng truy xuất của nhóm

> Cách chấm (theo `docs/SCORING.md`): **2 điểm/câu** — top-3 chứa chunk liên quan + agent trả lời đúng (2), có liên quan nhưng thiếu/không ở top-1 (1), không có trong top-3 (0).

| # | Câu hỏi | Chiến lược tốt nhất cho câu này | Có chunk liên quan trong top-3? | Ghi chú |
|---|---------|-------------------------------|-------------------------------|---------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

**Lọc bằng metadata có giúp ích không? Ở câu hỏi nào?**
> *Viết 2-3 câu:*

---

## 4. Thuyết trình (Demo) & Bài học nhóm — Nhóm (5 điểm)

**Những phân tích (insights) hay nhất nhóm sẽ trình bày:**
> *Liệt kê 2-3 ý:*

**Bài học rút ra khi so sánh trong nhóm:**
> *Viết 2-3 câu — cùng tài liệu nhưng chiến lược khác nhau dẫn tới khác biệt gì?*

**Nếu làm lại, nhóm sẽ thay đổi gì trong chiến lược dữ liệu (data strategy)?**
> *Viết 2-3 câu:*

---

## Tự Đánh Giá (Phần Nhóm)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Lựa chọn tài liệu (Document Set Quality) | 10 / 10 |
| Thiết kế chiến lược (Strategy Design) | / 15 |
| Chất lượng truy xuất (Retrieval Quality) | / 10 |
| Thuyết trình (Demo) | / 5 |
| **Tổng phần nhóm** | **/ 40** |
