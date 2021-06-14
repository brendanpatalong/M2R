  
mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"bp419@ic.ac.uk\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml