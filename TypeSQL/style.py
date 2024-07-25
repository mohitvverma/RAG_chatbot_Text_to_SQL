import streamlit as st


def markdown_style() -> None:
    """
    setup the markdown style for Streamlit UI
    :return:
    """
    st.markdown("""
                <style>
                # MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """, unsafe_allow_html=True
                )

    st.markdown(
        """
        <style>.st-emotion-cache-janbn0 {
        flex-direction: row-reverse;
        text-align: right;
        }
        </style>
        """,
        unsafe_allow_html=True)

    st.markdown(
        """
        <style>
            .eeusbqq2 {
            margin-top: 8px;
            margin-left: -5px;
            width: 35px;
            height: 35px;
            }
            .e16edly10 {
                font-size: 20px;
            }
        <style>
        """,
        unsafe_allow_html=True
    )