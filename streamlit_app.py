"""Root entrypoint for the Streamlit app."""

def main() -> None:
    from app.ui.streamlit_app import main as app_main

    app_main()


if __name__ == "__main__":
    main()
