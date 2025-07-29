import streamlit as st
import time
import datetime
import tempfile
import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Check if we have API keys configured


def check_api_keys():
    openai_key = os.getenv("OPENAI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    return openai_key is not None or groq_key is not None


# Constants
APP_VERSION = "v2.0.0-Enhanced"

# Grant templates
GRANT_TEMPLATES = {
    "Research Grant": {
        "description": "Academic or scientific research funding application",
        "sections": ["Executive Summary", "Research Background", "Methodology", "Expected Outcomes", "Budget", "Timeline"],
        "prompt": "A research grant for academic or scientific investigation"
    },
    "Non-Profit Program": {
        "description": "Funding for non-profit organization programs",
        "sections": ["Organization Overview", "Program Description", "Community Impact", "Sustainability Plan", "Budget", "Evaluation Metrics"],
        "prompt": "A non-profit program grant focused on community impact"
    },
    "Educational Initiative": {
        "description": "Funding for educational programs or institutions",
        "sections": ["Project Summary", "Educational Objectives", "Implementation Plan", "Target Population", "Budget", "Assessment Plan"],
        "prompt": "An educational initiative grant for learning programs"
    },
    "Technology Innovation": {
        "description": "Funding for technological innovation or product development",
        "sections": ["Innovation Overview", "Market Analysis", "Technical Approach", "Commercialization Plan", "Budget", "Team Qualifications"],
        "prompt": "A technology innovation grant for developing new solutions"
    },
    "Healthcare Project": {
        "description": "Funding for healthcare-related projects",
        "sections": ["Project Summary", "Healthcare Need", "Implementation Strategy", "Expected Health Outcomes", "Budget", "Evaluation Plan"],
        "prompt": "A healthcare project grant addressing medical needs"
    },
    "Custom": {
        "description": "Create a custom grant proposal structure",
        "sections": [],
        "prompt": ""
    }
}


def main():
    st.set_page_config(
        page_title="Enhanced Grant Writing Assistant",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🚀 Enhanced AI Grant Writing Assistant")
    st.subheader("Multi-Mode Grant Proposal Generation with Advanced Features")

    # Check for API keys
    has_api_keys = check_api_keys()

    # Sidebar
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/000000/document-writer.png", width=80)
        st.title("Grant Assistant")

        # API Key Status
        if has_api_keys:
            st.success("✅ API Keys Configured")
        else:
            st.error("❌ No API Keys Found")
            st.info("Please add OPENAI_API_KEY or GROQ_API_KEY to your .env file")

        # Mode selection
        app_mode = st.radio(
            "Mode", ["Standard", "Features Demo", "Setup Guide"])

    if app_mode == "Standard":
        if not has_api_keys:
            st.error("⚠️ **API Keys Required**")
            st.write(
                "Please configure your API keys in the .env file to use the grant generation features:")
            st.code("""
# Create or edit .env file in your project directory
OPENAI_API_KEY=your_openai_key_here
# OR
GROQ_API_KEY=your_groq_key_here
            """)
            return

        try:
            # Import modules only when API keys are available
            from crew_agents.crew_builder import build_crew
            from utils.doc_generator import save_to_word, save_to_pdf

            # Grant template selection
            selected_template = st.selectbox(
                "Select Grant Type",
                options=list(GRANT_TEMPLATES.keys()),
                format_func=lambda x: f"{x} - {GRANT_TEMPLATES[x]['description']}"
            )

            # Show template details
            if selected_template != "Custom":
                with st.expander("📋 Template Details"):
                    st.write(
                        f"**Description:** {GRANT_TEMPLATES[selected_template]['description']}")
                    st.write("**Sections:**")
                    for section in GRANT_TEMPLATES[selected_template]['sections']:
                        st.write(f"- {section}")

            # Grant topic input
            grant_topic = st.text_area(
                "Enter Grant Purpose or Topic:",
                placeholder=f"Describe your {selected_template.lower()} idea here...",
                height=100
            )

            # File upload
            uploaded_file = st.file_uploader(
                "Upload related PDF document (optional)", type=["pdf"])

            # Options
            col1, col2 = st.columns(2)
            with col1:
                use_template = st.checkbox(
                    "Use Template Structure", value=True)
            with col2:
                add_metadata = st.checkbox("Include Version Info", value=True)

            # Generate button
            if st.button("Generate Proposal", type="primary", use_container_width=True) and grant_topic.strip():
                with st.spinner("🤖 Generating proposal with AI agents..."):
                    try:
                        # Prepare for generation
                        temp_path = None
                        if uploaded_file:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                tmp.write(uploaded_file.read())
                                temp_path = tmp.name

                        # Build crew with enhanced prompt
                        enhanced_topic = grant_topic
                        if use_template and selected_template != "Custom":
                            template_info = GRANT_TEMPLATES[selected_template]
                            enhanced_topic += f"\n\nTemplate Type: {selected_template}"
                            enhanced_topic += f"\nRequired Sections: {', '.join(template_info['sections'])}"

                        crew = build_crew(
                            grant_topic=enhanced_topic,
                            pdf_path=temp_path
                        )

                        # Create progress indicator
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        # Simulate progress (in real implementation, this would be connected to CrewAI callbacks)
                        for i, step in enumerate(["Researching...", "Analyzing ideas...", "Writing draft...", "Formatting...", "Proofreading..."]):
                            status_text.text(f"Step {i+1}/5: {step}")
                            progress_bar.progress((i+1)/5)
                            time.sleep(1)

                        # Run crew
                        result = crew.kickoff()

                        if result is not None:
                            # Extract output text
                            output_text = result.output if hasattr(
                                result, "output") else str(result)

                            # Clear progress indicators
                            progress_bar.empty()
                            status_text.empty()

                            # Success message
                            st.success("✅ Proposal Generated Successfully!")

                            # Display proposal with metrics
                            word_count = len(output_text.split())
                            st.metric("Word Count", f"{word_count:,}")

                            # Display proposal
                            with st.expander("📄 Generated Grant Proposal", expanded=True):
                                st.text_area("", output_text, height=400)

                            # Prepare timestamp and metadata
                            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            metadata = f"{APP_VERSION}"
                            if add_metadata:
                                metadata += f" | Generated on {timestamp}"
                                metadata += f" | Template: {selected_template}"
                                metadata += f" | Words: {word_count:,}"

                            # Download options
                            st.subheader("📥 Download Options")
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.download_button(
                                    "📄 Download Word",
                                    save_to_word(output_text, metadata),
                                    file_name=f"grant_proposal_{timestamp.replace(':', '-').replace(' ', '_')}.docx",
                                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                                )

                            with col2:
                                st.download_button(
                                    "📑 Download PDF",
                                    save_to_pdf(output_text, metadata),
                                    file_name=f"grant_proposal_{timestamp.replace(':', '-').replace(' ', '_')}.pdf",
                                    mime="application/pdf"
                                )

                            with col3:
                                # Copy to clipboard (simplified)
                                if st.button("📋 Copy Text"):
                                    st.write(
                                        "💡 Use Ctrl+A then Ctrl+C to copy the proposal text above")
                        else:
                            st.error(
                                "Could not generate the proposal. Please try again.")

                    except Exception as e:
                        st.error(f"❌ Error generating proposal: {str(e)}")
                        st.info(
                            "💡 Try checking your API key configuration or try again in a moment.")

        except ImportError as e:
            st.error(f"❌ Import Error: {str(e)}")
            st.info(
                "Some modules may be missing. Please run: pip install -r requirements_enhanced.txt")

    elif app_mode == "Features Demo":
        st.write("### 🎯 Enhanced Features Overview")
        st.write("This demonstrates what the full enhanced version includes:")

        # Feature comparison
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🧙‍♂️ Wizard Mode")
            st.write("**Step-by-step guided process:**")
            st.write("1. 📋 Project Information")
            st.write("2. 🏢 Organization Details")
            st.write("3. 🎯 Goals & Objectives")
            st.write("4. 💰 Budget & Timeline")
            st.write("5. ⚡ Generate & Review")

            st.subheader("📊 Analytics Dashboard")
            st.write("**Track your proposals:**")
            st.write("- 📈 Word count trends over time")
            st.write("- 🥧 Section distribution analysis")
            st.write("- 📚 Complete proposal history")
            st.write("- 📊 Interactive charts & metrics")

        with col2:
            st.subheader("💾 Persistent Storage")
            st.write("**Never lose your work:**")
            st.write("- 💾 Auto-save functionality")
            st.write("- 📁 Draft management system")
            st.write("- 🔄 Backup & restore")
            st.write("- 📤 Export/import all data")

            st.subheader("🤖 Enhanced AI")
            st.write("**Advanced AI integration:**")
            st.write("- 🔄 Multiple AI providers (OpenAI, Groq)")
            st.write("- ⚙️ Dynamic model switching")
            st.write("- 📊 Real-time progress tracking")
            st.write("- 🛡️ Better error handling")

        # Show sample data visualization
        try:
            import pandas as pd
            import plotly.express as px

            st.subheader("📈 Sample Analytics")
            sample_data = pd.DataFrame({
                "Date": ["2025-07-20", "2025-07-22", "2025-07-24", "2025-07-26"],
                "Proposal": ["Research Grant", "Non-Profit Program", "Tech Innovation", "Healthcare Project"],
                "Word Count": [2500, 3200, 2800, 3500]
            })

            fig = px.line(sample_data, x="Date", y="Word Count",
                          hover_data=["Proposal"], markers=True,
                          title="Word Count Trends Over Time")
            st.plotly_chart(fig, use_container_width=True)

            # Pie chart for sections
            sections_data = pd.DataFrame({
                "Section": ["Executive Summary", "Problem Statement", "Methodology", "Budget", "Timeline"],
                "Words": [300, 800, 1200, 400, 300]
            })

            fig2 = px.pie(sections_data, values="Words", names="Section",
                          title="Latest Proposal Section Distribution")
            st.plotly_chart(fig2, use_container_width=True)

        except ImportError:
            st.info(
                "📊 Install plotly and pandas to see sample visualizations: pip install plotly pandas")

    else:  # Setup Guide
        st.write("### 🛠️ Setup Guide")

        st.subheader("1. 🔑 API Key Configuration")
        st.write("Add your API keys to the .env file in your project directory:")

        # Show current .env status
        env_path = ".env"
        if os.path.exists(env_path):
            st.success("✅ .env file found")
        else:
            st.warning("⚠️ No .env file found")

        st.code("""
# Create or edit .env file in your project directory
OPENAI_API_KEY=sk-your_openai_api_key_here
# OR
GROQ_API_KEY=gsk_your_groq_api_key_here

# You can use either provider or both
        """)

        st.subheader("2. 📦 Dependencies")
        st.write("Install required packages:")

        st.code("pip install -r requirements_enhanced.txt")

        # Show what's installed
        try:
            import streamlit as st_check
            import crewai
            st.success("✅ Core packages available")
        except ImportError:
            st.error("❌ Missing core packages")

        try:
            import plotly
            import pandas
            st.success("✅ Enhanced features packages available")
        except ImportError:
            st.warning("⚠️ Enhanced features packages not installed")

        st.subheader("3. 🚀 Usage")
        st.write("**Current Demo Version:**")
        st.code("streamlit run enhanced_grant_app_demo.py")

        st.write("**Your Original App:**")
        st.code("streamlit run streamlit2_app2.py")

        st.subheader("4. 🆚 Comparison")
        comparison_data = {
            "Feature": ["Basic Generation", "Templates", "PDF Upload", "Export Options", "UI Design", "Error Handling"],
            "Original App": ["✅", "❌", "✅", "Word/PDF", "Basic", "Basic"],
            "Enhanced Demo": ["✅", "✅ (5 types)", "✅", "Word/PDF + Metadata", "Modern", "Improved"],
            "Full Enhanced": ["✅", "✅ (5 types)", "✅", "Word/PDF + Analytics", "Multi-mode", "Advanced"]
        }

        import pandas as pd
        df = pd.DataFrame(comparison_data)
        st.table(df)

        st.subheader("5. 🎯 Next Steps")
        st.write("1. ✅ Configure your API keys")
        st.write("2. ✅ Test the enhanced demo")
        st.write("3. ✅ Compare with your original app")
        st.write("4. ✅ Choose features you want to implement fully")


if __name__ == "__main__":
    main()
