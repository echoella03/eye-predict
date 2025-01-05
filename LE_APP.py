import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models

# Prediction Labels
labels = [
    "Central Serous Chorioretinopathy", "Diabetic Retinopathy", "Disc Edema", 
    "Glaucoma", "Healthy", "Macular Scar", "Myopia", "Pterygium", 
    "Retinal Detachment", "Retinitis Pigmentosa"
]

disease_details = {
    "Central Serous Chorioretinopathy": (
        "Central Serous Chorioretinopathy (CSCR) is an eye condition caused by the buildup of fluid under the retina, "
        "leading to detachment and visual distortion. It is often associated with stress or steroid use."
    ),
    "Diabetic Retinopathy": (
        "Diabetic Retinopathy is caused by damage to the blood vessels in the retina due to prolonged high blood sugar levels. "
        "It progresses through stages, from mild non-proliferative changes to severe proliferative retinopathy."
    ),
    "Disc Edema": (
        "Disc Edema is the swelling of the optic nerve head, often caused by increased intracranial pressure or optic nerve inflammation. "
        "It may be associated with headaches, blurred vision, or double vision. "
    ),
    "Glaucoma": (
        "Glaucoma is a group of diseases characterized by damage to the optic nerve, typically caused by elevated intraocular pressure. "
        "It can lead to peripheral vision loss and, if untreated, blindness. "
    ),
    "Healthy": (
        "Healthy eyes show no signs of disease or abnormalities in structure or function. "
    ),
    "Macular Scar": (
        "Macular Scar is scarring in the central part of the retina (macula), often resulting from injury, inflammation, or advanced retinal conditions. "
        "It can significantly impair central vision. "
    ),
    "Myopia": (
        "Myopia, or nearsightedness, is a refractive error where distant objects appear blurry while nearby objects are clear. "
        "It occurs due to the elongation of the eyeball or excessive curvature of the cornea. "
    ),
    "Pterygium": (
        "Pterygium is a non-cancerous growth of the conjunctiva that can extend onto the cornea, often caused by prolonged UV exposure. "
        "It may cause irritation, redness, or blurry vision. "
    ),
    "Retinal Detachment": (
        "Retinal Detachment occurs when the retina separates from its underlying support tissue, leading to vision loss if untreated. "
        "It can result from trauma, high myopia, or retinal tears. "
    ),
    "Retinitis Pigmentosa": (
        "Retinitis Pigmentosa is a genetic disorder causing progressive loss of photoreceptor cells, beginning with night blindness and peripheral vision loss. "
    )
}

# Disease Details
other = {
    "Central Serous Chorioretinopathy": (
        "Under retinal imaging, there may be a blister-like swelling in the central retina (macula), "
        "visible as a dome-shaped elevation."
    ),
    "Diabetic Retinopathy": (
        "Retinal imaging shows microaneurysms, hemorrhages, cotton wool spots, and in advanced stages, "
        "new blood vessel growth (neovascularization)."
    ),
    "Disc Edema": (
        "The optic disc appears swollen and blurred with possible hemorrhages around the disc margin."
    ),
    "Glaucoma": (
        "Optic nerve imaging shows cupping (enlarged optic cup), thinning of the neuroretinal rim, and a pale disc."
    ),
    "Healthy": (
        "The retina, optic nerve, and other structures appear normal with no lesions, swelling, or abnormal growths."
    ),
    "Macular Scar": (
        "The macula may have a grayish or white appearance with fibrous tissue formation, visible on fundus imaging."
    ),
    "Myopia": (
        "Fundus images may show tilted optic discs, a myopic crescent near the optic disc, and chorioretinal thinning in severe cases."
    ),
    "Pterygium": (
        "A wedge-shaped, fleshy growth is visible on the white part of the eye (sclera) and may invade the cornea."
    ),
    "Retinal Detachment": (
        "Retinal imaging shows the retina lifted away from the back of the eye, often with folds or a ballooned appearance."
    ),
    "Retinitis Pigmentosa": (
        "Retinal imaging reveals a classic pattern of black, bone-spicule-like pigment deposits, attenuated blood vessels, and pale optic discs."
    )
}

# Load your pretrained DenseNet model
@st.cache_resource
def load_model():
    model = models.densenet201(pretrained=False)
    num_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(num_features, len(labels))  # Update for 10 classes
    model.load_state_dict(torch.load("densenet201_2_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocess the image
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)

# Predict the disease
def predict(image, model):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)  # Apply softmax to get probabilities
        confidence, predicted = torch.max(probabilities, 1)
    return predicted.item(), confidence.item(), probabilities.squeeze().tolist()

# Streamlit app configuration
st.set_page_config(page_title="Eye Disease Classification", layout="wide")

# Sidebar for navigation
with st.sidebar:
    st.markdown(
        """
        <h1 style="text-align: center; font-size: 35px; font-weight: bold; color: #06233B; font-family: 'Impact', sans-serif;">EYE SEE U</h1>
        """,
        unsafe_allow_html=True,
    )
    if "active_tab" not in st.session_state:
        st.session_state["active_tab"] = "Home"  # Default to Home
    st.session_state["active_tab"] = st.radio("", ["Home", "Predict", "About"], index=["Home", "Predict", "About"].index(st.session_state["active_tab"]))

# Set the active tab based on session state
active_tab = st.session_state["active_tab"]


# Home Tab
if active_tab == "Home" or active_tab is None:
    # Center align the content using Streamlit's layout
    col1, col2, col3, col4 = st.columns([2, 2, 5, 1])  # Create columns for centering content

    with col2:  # Column for the logo
        st.image("assets/slant_logo.png", use_container_width=True)

    with col3:  # Column for the title
        st.markdown(
            """
            <h1 style="text-align: left; font-size: 90px; font-weight: bold; color: #06233B; font-family: 'Impact', sans-serif;">EYE SEE U</h1>
            """,
            unsafe_allow_html=True,
        )
        
        
    # Subtitle and description
    st.markdown(
        """
        <h3 style="text-align: center; font-size: 24px; color: #555;">AI-Powered Retinal Disease Diagnosis</h3>
        <p style="text-align: center; font-size: 18px; max-width: 1500px; margin: auto; color: #333;">
            This is a cutting-edge application that leverages deep learning techniques 
            to assist ophthalmologists in diagnosing retinal diseases. By analyzing eye images, 
            our system can provide accurate predictions and details about various eye conditions.
        </p>
        """,
        unsafe_allow_html=True,
    )


    st.image("assets/ai.jpg", use_container_width=True)

    # Trivia and facts section
    st.markdown(
            """
            <h1 style="text-align: center; font-size: 35px; font-weight: bold; color: #06233B; font-family: 'Impact', sans-serif;">DO YOU KNOW?</h1>
            """,
            unsafe_allow_html=True,
        )

    # Create frames for trivia and facts
    fact1, fact2, fact3, fact4 = st.columns(4)  # Three equal-sized columns for facts

    with fact1:
        st.markdown(
            """
            <div style="border: 1px solid #ddd; padding: 15px; border-radius: 10px; background-color: #F3F8FB; 
                        width: 220px; height: 220px; margin: auto;">
                <p style="text-align: center; font-size: 18px; color: #06233B;">Human eye can differentiate approximately 10 million colors 
                and has a resolution equivalent to about 576 megapixels.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with fact2:
        st.markdown(
            """
            <div style="border: 1px solid #ddd; padding: 15px; border-radius: 10px; background-color: #F3F8FB; 
                        width: 220px; height: 220px; margin: auto;">
                <p style="text-align: center; font-size: 18px; color: #06233B;">Eye diseases are very common, with over 2.2 billion people 
                worldwide experiencing vision impairment or blindness.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with fact3:
        st.markdown(
            """
            <div style="border: 1px solid #ddd; padding: 15px; border-radius: 10px; background-color: #F3F8FB; 
                        width: 220px; height: 220px; margin: auto;">
                <p style="text-align: center; font-size: 18px; color: #06233B;">Excessive exposure to blue light from screens can lead to 
                digital eye strain or computer vision syndrome.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with fact4:
        st.markdown(
            """
            <div style="border: 1px solid #ddd; padding: 15px; border-radius: 10px; background-color: #F3F8FB; 
                        width: 220px; height: 220px; margin: auto;">
                <p style="text-align: center; font-size: 18px; color: #06233B;">The muscles that control your eyes are among the most active 
                and fastest in the human body.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    

# Predict Tab
if active_tab == "Predict":
  # Create two columns: one for the logo and one for the title       
    col3, col4 = st.columns([1, 10])  # Adjust column width ratios as needed

        # Add the logo to the first column
    with col3:
        st.image("assets/slant_logo.png", width=500)  # Adjust path and size as needed

        # Add the title to the second column
    with col4:
        st.markdown(
            """
            <h1 style="text-align: left; font-size: 50px; color: #06233B; font-family: 'Franklin Gothic Heavy', sans-serif;">DIAGNOSE RETINAL IMAGES</h1>
            """,
            unsafe_allow_html=True,
        )

    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload Your Eye Image")
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.subheader("Prediction Results")
        if uploaded_file:
            # Create a container for the results
            with st.container():
                model = load_model()
                predicted_class_index, confidence, probabilities = predict(image, model)
                predicted_class = labels[predicted_class_index]
                
                # Center-align the results using markdown and CSS
                st.markdown(
                    """
                    <div style="text-align: center;">
                        <p><strong>Predicted Class:</strong> {}</p>
                        <p><strong>Confidence Level:</strong> {:.2f}%</p>
                        <hr>
                        <h4>Details</h4>
                        <p>{}</p>
                        <p><h4>Characteristic:</h4> {}</p>
                    </div>
                    """.format(
                        predicted_class,
                        confidence * 100,
                        disease_details.get(predicted_class, "No detailed information available for this condition."),
                        other.get(predicted_class, "No additional characteristics available.")
                    ),
                    unsafe_allow_html=True
                )
                

# About Tab
if active_tab == "About":
    st.markdown(
        """
        <style>
        /* General Container Styling */
        .header-container {
            text-align: center;
            padding: 30px 0;
        }
        .header-title {
            font-size: 40px;
            font-weight: bold;
            color: #06233B;
        }
        .header-subtitle {
            font-size: 18px;
            color: #15588F;
            font-style: italic;
        }
        .section-container {
            background-color: #F3F8FB;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .section-title {
            font-size: 30px;
            font-weight: bold;
            color: #15588F;
            margin-bottom: 20px;
        }
        .team-member {
            display: flex;
            align-items: center;
            gap: 20px;
            margin-bottom: 30px;
        }
        .team-member img {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            border: 2px solid #15588F;
        }
        .team-member-details {
            font-size: 18px;
            color: #06233B;
        }
        .team-member-role {
            font-size: 20px;
            font-weight: bold;
            color: #15588F;
            margin-top: 5px;
        }
        .contact-info {
            text-align: center;
            font-size: 18px;
            color: #15588F;
            margin-top: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 10])  # Adjust column width ratios as needed

    # Add the logo to the first column
    with col1:
        st.image("assets/slant_logo.png", width=450)  # Adjust path and size as needed

    # Add the title to the second column
    with col2:
        st.markdown(
            """
            <h1 style="text-align: left; font-size: 50px; color: #06233B; font-family: 'Franklin Gothic Heavy', sans-serif;">ABOUT</h1>
            """,
            unsafe_allow_html=True,
        )
    # About Section
    st.markdown(
        '<div class="section-container"><p style="font-size:20px;">Eye See U is a research-based AI system designed to revolutionize eye health assessment. By leveraging advanced deep learning techniques, specifically Convolutional Neural Networks (CNNs), we aim to provide accurate and timely diagnoses of various eye conditions.</p></div>',
        unsafe_allow_html=True,
    )

    # Why Choose Us Section
    st.markdown(
        """
        <div class="section-container">
            <div class="section-title">Why Choose Eye See U?</div>
            <ul style="font-size:18px; color:#06233B;">
                <li><b>Accuracy:</b> Our AI model is trained on a vast dataset of eye images, ensuring high accuracy in disease classification.</li>
                <li><b>Efficiency:</b> Quick and convenient assessments, eliminating the need for immediate in-person consultations.</li>
                <li><b>Early Detection:</b> Early detection of eye diseases can lead to timely intervention and improved outcomes.</li>
                <li><b>Accessibility:</b> Our platform is accessible to people worldwide, providing eye health assessments to those who may not have easy access to healthcare facilities.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Know the Team Section
    st.title("KNOW THE TEAM")
    
    col1, col2 = st.columns(2)  # Adjust column width ratios as needed

    # Add the logo to the first column
    with col1:
        # Team Member - Daniela J. Comapon
        st.markdown('<div class="team-member">', unsafe_allow_html=True)
        st.image("assets/dan.jpg", caption="DANIELA J. COMAPON", use_container_width=True)
        st.markdown(
            """
            <div class="section-container" style="padding: 30px; text-align: center;">
                <div class="team-member-details" style="text-align: center;">
                    <div class="team-member-role" style="font-size: 20px; font-weight: bold; color: #06233B;">UI/Programmer</div>
                    <p style="font-size: 16px; color: #333;">"Her task is to debug and fix the system, ensuring they meet the specified requirements and function effectively."</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown('</div>', unsafe_allow_html=True)
        
    # Add the title to the second column
    with col2:
        # Team Member - Andrae D. Bretana
        st.markdown('<div class="team-member">', unsafe_allow_html=True)
        st.image("assets/drae.jpg", caption="ANDRAE D. BRETANA", use_container_width=True)
        st.markdown(
            """
            <div class="section-container" style="padding: 30px; text-align: center;">
                <div class="team-member-details" style="text-align: center;">
                    <div class="team-member-role" style="font-size: 20px; font-weight: bold; color: #06233B;">Project Leader</div>
                    <p style="font-size: 16px; color: #333;">"His role is to oversee the system's overall performance and address any issues that arise."</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown('</div>', unsafe_allow_html=True)



    # Contact Us Section
    st.markdown(
        """
        <div class="section-container">
            <div class="contact-info">
                <p><b>Contact us:</b></p>
                <p>Gmail: djcomapon02137@usep.edu.ph | adbretana03103@usep.edu.ph</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
