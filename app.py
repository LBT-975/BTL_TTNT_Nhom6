import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt  
from unidecode import unidecode
from sklearn.base import BaseEstimator, TransformerMixin

# 1. ĐỊNH NGHĨA CLASS VÀ HÀM TRƯỚC
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, preprocess_func):
        self.preprocess_func = preprocess_func

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # ... code của bạn ...
        return X

def full_preprocess(text):
    return str(text)

# 2. SAU ĐÓ MỚI LOAD MODEL
@st.cache_resource
def load_model():
    return joblib.load("tfidf_svm_pipeline.pkl")

model = load_model()

# NHÃN TIẾNG VIỆT
label_vi = {
    "POS": "Tích cực 😊",
    "NEG": "Tiêu cực 😠",
    "NEU": "Trung lập 😐"
}


# ĐỘ TIN CẬY
def confidence_level(prob):
    if prob >= 0.5:
        return "Cao 🔥"
    elif prob >= 0.3:
        return "Trung bình ⚖️"
    else:
        return "Thấp ⚠️"


def normalize_text(text):
    text = str(text).lower()
    text = unidecode(text)

    # loại bỏ khoảng trắng thừa
    text = re.sub(r"\s+", " ", text)

    # spam ký tự
    text = re.sub(r"(.)\1{2,}", r"\1", text)


    replacements = {
    "ko": "khong",
    "k ": "khong ",
    "hok": "khong",
    "khum": "khong",
    "sp": "san pham",
    "nv": "nhan vien",
    "shoppe": "shopee",
    "bth": "binh thuong",
    "bt": "binh thuong",
    "okela": "ok",
    "okee": "ok",
    "dc": "duoc",
    "đc": "duoc"
    }


    for old, new in replacements.items():
        text = text.replace(old, new)


    return text.strip()


def rule_based_sentiment(comment):

    text = normalize_text(comment)

    # kiểm tra emoji trên comment gốc
    if any(e in comment for e in positive_emojis):
        return "POS"
    if any(e in comment for e in negative_emojis):
        return "NEG"

    # Đếm sao trước
    star_count = comment.count("⭐")

    if star_count >= 4:
        return "POS"
    elif star_count == 3:
        return "NEU"
    elif 0 < star_count <= 2:
        return "NEG"
   
    # Trung lập trước
    if "khong qua te nhung cung khong tot" in text:
        return "NEU"
    if "khong tot cung khong xau" in text:
        return "NEU"
    if "giao hang dung hen" in text:
        return "POS"
   
    # Phủ định tiêu cực
    negative_phrases = [
    "khong tot",
    "khong dep",
    "khong on",
    "khong hai long",
    "khong dang tien",
    "khong thich",
    "khong mua lai"
    ]

    if any(p in text for p in negative_phrases):
        return "NEG"

    # Phủ định tích cực
    positive_phrases = [
    "khong den noi",
    "khong toi"
    ]

    if any(p in text for p in positive_phrases):
        return "POS"

    # Câu hỗn hợp
    for word in contrast_words:

     if word in text:

        left, right = text.split(word, 1)

        left_pos = any(k in left for k in positive_keywords)
        left_neg = any(k in left for k in negative_keywords)

        right_pos = any(k in right for k in positive_keywords)
        right_neg = any(k in right for k in negative_keywords)

        # Vế sau quan trọng hơn

        if right_neg:
            return "NEG"

        if right_pos:
            return "POS"

        return "NEU"

    # Keyword trung lập
    if any(w in text for w in neutral_keywords):
        return "NEU"
   
    if any(word in text for word in uncertain_keywords):
       return "NEU"

    # Keyword tiêu cực
    if any(w in text for w in negative_keywords):
        return "NEG"

    # Keyword tích cực    
    if any(w in text for w in positive_keywords):
        return "POS"
   
    if any(x in text for x in special_neutral):
      return "NEU"
   
    if "nhung" in text:

     before, after = text.split("nhung", 1)

     pos_after = any(x in after for x in positive_keywords)
     neg_after = any(x in after for x in negative_keywords)

     if neg_after:
        return "NEG"

     if pos_after:
        return "POS"

     return "NEU"

    return None


# TỪ KHÓA HỖ TRỢ
negative_keywords = list(set([
    "qua te", "rat te", "te lam", "cuc te", "that vong", "khong hai long",
    "kem chat luong", "khong mua lai", "thai do loi lom", "thai do kem",
    "dich vu kem", "chu thoi", "thui", "toi te", "khong tot", "khong on",
    "khong dang tien", "chat luong kem", "giao hang cham", "dong goi so sai",
    "gia cao", "mac", "qua mac", "khong dang mua", "khong nhu mong doi",
    "khong giong hinh", "hang gia", "lua dao", "chan", "bo tay",
    "khong thich", "san pham bi loi", "hang bi hu", "giao sai hang",
    "giao thieu hang", "phuc vu te", "dich vu te", "thai do te", "te",
    "kha te", "chat luong te", "chat luong kha te"
]))

positive_keywords = list(set([
    "rat tot", "tot lam", "tot qua", "tuyet voi", "xuat sac", "hoan hao",
    "hai long", "rat hai long", "cuc ky hai long", "dang tien", "rat dang tien",
    "chat luong tot", "chat luong cao", "giao hang nhanh", "dong goi can than",
    "phuc vu tot", "dich vu tot", "nhan vien nhiet tinh", "nhan vien than thien",
    "tu van tan tam", "cham soc khach hang tot", "se mua lai", "se ung ho tiep",
    "se quay lai", "rat thich", "yeu thich", "ung y", "rat ung",
    "ngoai mong doi", "vuot mong doi", "5 sao", "10 diem",
    "khong co gi de che", "giao hang dung hen", "chat luong vuot mong doi",
    "ok", "oke", "good", "dep", "xin", "xin so", "on ap", "dang mua",
    "shop co tam", "tot", "kha tot", "chat luong kha tot"
]))

positive_emojis = list(set(["😍","🥰","❤️","👍","👏","🔥","🤩","😊","😁","😄","😆","🥳"]))

neutral_keywords = list(set([
    "binh thuong", "toi thay binh thuong", "tam duoc", "cung duoc", "kha on",
    "tam tam", "o muc chap nhan duoc", "khong co gi dac biet",
    "khong co gi noi bat", "khong qua te nhung cung khong tot",
    "k qua te nhung cung k tot", "ko qua te nhung cung ko tot",
    "khong tot cung khong xau", "k tot cung k xau", "ko tot cung ko xau",
    "dong goi binh thuong", "giao hang binh thuong", "dich vu binh thuong",
    "chat luong binh thuong", "o muc trung binh", "chua dung",
    "chua danh gia duoc", "moi nhan hang", "moi nhan duoc",
    "dang trai nghiem", "can them thoi gian", "chua su dung",
    "chua biet chat luong", "moi mo hop", "de dung thu xem sao"
]))

uncertain_keywords = list(set([
    "khong biet sao", "dang suy nghi", "can nhac", "xem them", "tham khao",
    "cho them review", "cho xem them", "khong chac", "chua chac",
    "chua dam mua", "van phan van", "dang lan tan", "hinh nhu",
    "co le", "khong ro", "khong biet chat luong ra sao"
]))

negative_emojis = list(set(["😡","😠","👎","🤮","💔"]))

negation_words = list(set(["khong","ko","k","kh","chang","chua"]))

contrast_words = list(set(["nhung","tuy nhien","mac du","du"]))

special_neutral = list(set(["khong te nhung cung khong tot", "khong tot cung khong xau","tam on","tam tam"]))

result = "NEU"

# GIAO DIỆN
st.set_page_config(
    page_title="Phân tích cảm xúc bình luận đánh giá sản phẩm",
    page_icon="💬",
    layout="centered"
)

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #eef2ff 0%,
        #f8fafc 50%,
        #dbeafe 100%
    );
}

.main {
    padding-top: 1rem;
}

h1 {
    text-align:center;
}

.stButton>button {
    width:100%;
    border-radius:15px;
    height:55px;
    font-size:18px;
    font-weight:bold;
}

div[data-testid="stMetric"] {
    background:white;
    padding:15px;
    border-radius:15px;
    box-shadow:0 4px 10px rgba(0,0,0,0.1);
}

.result-box{
    box-shadow:0 8px 20px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<h1 class='title-center'>
💬 HỆ THỐNG PHÂN TÍCH CẢM XÚC BÌNH LUẬN
</h1>
<p style='text-align:center'>
Ứng dụng AI phân tích đánh giá sản phẩm bằng TF-IDF + SVM
</p>
""", unsafe_allow_html=True)


# =========================
# PHÂN TÍCH 1 BÌNH LUẬN
# =========================
   
comment = st.text_area(
    "✍️ Nhập bình luận:",
    placeholder="Ví dụ: Sản phẩm rất tốt, giao hàng nhanh..."
)

if st.button("🔍 Phân tích"):


    if not comment.strip():
        st.warning("⚠️ Vui lòng nhập bình luận!")
    else:

        pred = model.predict([comment])[0]
        result = rule_based_sentiment(comment)

        if result is None:
          result = pred

        comment = comment.replace("ko", "không")
        comment = comment.replace("k ", "không ")
        comment = comment.replace("hok", "không")
        comment = comment.replace("sp", "sản phẩm")

        scores = model.decision_function([comment])[0]
        scores = np.array(scores)

        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / np.sum(exp_scores)

        labels = list(model.classes_)

        best_prob = float(np.max(probs))

        st.subheader("💬 Bình luận đã nhập")
        st.info(comment)

        st.subheader("🎯 Kết quả")

        if result == "POS":
          st.markdown("""
          <div class='result-box'
            style='background:#d4edda;
                padding:20px;
                border-radius:15px;
                border-left:8px solid #28a745;'>
            <h2>😊 TÍCH CỰC</h2>
            <p>Khách hàng có phản hồi tích cực về sản phẩm hoặc dịch vụ.</p>
           </div>
            """, unsafe_allow_html=True)

        elif result == "NEG":
          st.markdown("""
          <div class='result-box'
            style='background:#f8d7da;
                padding:20px;
                border-radius:15px;
                border-left:8px solid #dc3545;'>
           <h2>😠 TIÊU CỰC</h2>
           <p>Khách hàng thể hiện sự không hài lòng đối với sản phẩm hoặc dịch vụ.</p>
          </div>
           """, unsafe_allow_html=True)

        else:
          st.markdown("""
          <div class='result-box'
            style='background:#fff3cd;
                padding:20px;
                border-radius:15px;
                border-left:8px solid #ffc107;'>
            <h2>😐 TRUNG LẬP</h2>
            <p>Khách hàng đưa ra nhận xét mang tính trung lập hoặc chưa thể hiện cảm xúc rõ ràng.</p>
          </div>
           """, unsafe_allow_html=True)
       
        st.subheader("📊 Đánh giá của hệ thống")

        st.write(
             f"🔥 Mức đánh giá: {confidence_level(best_prob)}"
         )
       
        if any(word in normalize_text(comment)
            for word in uncertain_keywords):

              st.info(
              "🤔 Bình luận thể hiện sự do dự hoặc chưa chắc chắn."
        )
   
        else:
           st.progress(best_prob)

           st.metric(
            label="Độ tin cậy",
            value=f"{best_prob*100:.1f}%"
       )
# =========================
# PHÂN TÍCH HÀNG LOẠT
# =========================
st.write("---")
st.header("📂 Phân tích hàng loạt từ file")

uploaded_file = st.file_uploader(
    "Tải lên file CSV hoặc Excel",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:

    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success(f"Đã tải thành công file với {len(df)} dòng dữ liệu")

        st.subheader("👀 Xem trước dữ liệu")
        st.dataframe(df.head())

        text_column = st.selectbox(
            "Chọn cột chứa bình luận",
            df.columns
        )

        if st.button("🚀 Phân tích file"):
            comments = df[text_column].fillna("").astype(str)

            results = []

            for comment in comments:

                pred = model.predict([comment])[0]

                rule_result = rule_based_sentiment(comment)

                if rule_result is not None:
                  pred = rule_result

                results.append(label_vi.get(pred, pred))

            df["Cảm xúc"] = results

            st.subheader("📋 Kết quả phân tích")
            st.dataframe(df)
            st.write(f"Tổng số bình luận: {len(df)}")

            pos_count = (df["Cảm xúc"] == "Tích cực 😊").sum()
            neg_count = (df["Cảm xúc"] == "Tiêu cực 😠").sum()
            neu_count = (df["Cảm xúc"] == "Trung lập 😐").sum()

            st.subheader("📊 Thống kê")

            col1, col2, col3 = st.columns(3)

            with col1:
              st.metric(
                 "😊 Tích cực",
                 pos_count
               )

            with col2:
             st.metric(
                "😠 Tiêu cực",
                neg_count
             )

            with col3:
             st.metric(
                "😐 Trung lập",
                neu_count
             )
            fig, ax = plt.subplots(figsize=(6,6))

            ax.pie(
              [pos_count, neg_count, neu_count],
              labels=["Tích cực", "Tiêu cực", "Trung lập"],
              autopct="%1.1f%%"
            )

            ax.set_title("Phân bố cảm xúc")

            st.pyplot(fig)

            csv = df.to_csv(
                index=False
            ).encode("utf-8-sig")

            st.download_button(
                "📥 Tải file kết quả",
                csv,
                "ket_qua_phan_tich.csv",
                "text/csv"
            )

    except Exception as e:
        st.error(f"Lỗi khi đọc file: {e}")

