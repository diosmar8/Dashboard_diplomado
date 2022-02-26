mkdir -p ~/.streamlit/
echo "
[theme]
base='light'
backgroundColor='#f3f3f3'
secondaryBackgroundColor='#418c4d'
font='monospace'
[server]
headless = true
enableCORS=false
enableXsrfProtection=false
port = $PORT
" > ~/.streamlit/config.toml

