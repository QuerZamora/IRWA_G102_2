'''
import os
from json import JSONEncoder
# pip install httpagentparser
import httpagentparser  
import nltk
from flask import Flask, render_template, session
from flask import request
import pickle
import httpagentparser  # for getting the user agent as json
import uuid
from myapp.analytics.analytics import AnalyticsData, ClickedDoc
from myapp.search.load_corpus import load_corpus
from myapp.search.objects import Document, StatsDocument, ResultItem
from myapp.search.search_engine import SearchEngine
from myapp.search.algorithms import create_index_tfidf
from datetime import datetime, timedelta
import pytz
from flask import escape
sanitized_query = escape(request.form['search-query'])



# *** for using method to_json in objects ***
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)


_default.default = JSONEncoder().default
JSONEncoder.default = _default

# instantiate the Flask application
app = Flask(__name__)

# random 'secret_key' is used for persisting data in secure cookie
app.secret_key = 'afgsreg86sr897b6st8b76va8er76fcs6g8d7'
# open browser dev tool to see the cookies
app.config['SESSION_COOKIE_NAME'] = 'IRWA_SEARCH_ENGINE'


# Function to save index using pickle
def save_index(index, tf, df, idf, filename="index.pkl"):
    with open(filename, "wb") as f:
        pickle.dump((index, tf, df, idf), f)
    print(f"Index saved to {filename}")

# Function to load index using pickle
def load_index(filename="index.pkl"):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            index, tf, df, idf = pickle.load(f)
        print(f"Index loaded from {filename}")
        return index, tf, df, idf
    else:
        print(f"No saved index found at {filename}. Recomputing...")
        return None
    
# print("current dir", os.getcwd() + "\n")
# print("__file__", __file__ + "\n")
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
# print(path + ' --> ' + filename + "\n")
# load documents corpus into memory.
file_path = os.path.join(path, "farmers-protest-tweets.json")

# file_path = "../../tweets-data-who.json"
# corpus = load_corpus(file_path)
#print("loaded corpus. first elem:", list(corpus.values())[0])

# Instantiate the search engine
search_engine = SearchEngine()

# instantiate our in memory persistence
analytics_data = AnalyticsData()

# Check if the index is already saved
index_data = load_index()

if os.path.exists("corpus.pkl"):
    # Cargar el corpus desde el archivo .pkl
    print("Cargando corpus desde archivo guardado...")
    with open("corpus.pkl", "rb") as pkl_file:
        corpus = pickle.load(pkl_file)
else:
    # Crear el corpus desde el archivo original
    print("Generando nuevo corpus...")
    corpus = load_corpus(file_path)

    # Guardar el corpus en un archivo .pkl
    with open("corpus.pkl", "wb") as pkl_file:
        pickle.dump(corpus, pkl_file)
    print("Corpus guardado en corpus.pkl")


if index_data is None:
    # If no saved index, create it
    print("Creating a new index...")
    index, tf, df, idf = create_index_tfidf(corpus)
    save_index(index, tf, df, idf)  # Save the new index
else:
    # Load the saved index
    index, tf, df, idf = index_data

# Home URL "/"
@app.route('/')
def index():
    print("starting home url /...")

    # flask server creates a session by persisting a cookie in the user's browser.
    # the 'session' object keeps data between multiple requests
    session['some_var'] = "IRWA 2024 home"

    user_agent = request.headers.get('User-Agent')
    print("Raw user browser:", user_agent)

    user_ip = request.remote_addr
    agent = httpagentparser.detect(user_agent)

    print("Remote IP: {} - JSON user browser {}".format(user_ip, agent))

    print(session)

    return render_template('index.html', page_title="Welcome")

@app.route('/search', methods=['POST'])
def search_form_post():
    search_query = request.form['search-query']

    session['last_search_query'] = search_query

    # Store the initiation time of the search in the session
    session['search_initiation_time'] = datetime.now(pytz.utc)

    search_id = analytics_data.save_query_terms(search_query)

    results = search_engine.search(search_query, search_id, corpus)

    found_count = len(results)
    session['last_found_count'] = found_count

    print(session)

    return render_template('results.html', results_list=results, page_title="Results", found_counter=found_count)

@app.route('/doc_details', methods=['GET'])
def doc_details():
    # getting request parameters:

    print("doc details session: ")
    print(session)

    res = session["some_var"]

    print("recovered var from session:", res)

    # get the query string parameters from request
    clicked_doc_id = request.args["id"]

    tweet = request.args["tweet"]
    username = request.args["username"]
    date = request.args["date"]
    likes = request.args["likes"]
    retweets = request.args["retweets"]
    url = request.args["url"]
    ranking = request.args["ranking"]
    ranking = round(float(ranking), 4)
    
    document = (ResultItem(
                id=clicked_doc_id,
                title=None,
                tweet=tweet,
                username=username,
                date=date,
                likes=likes,
                retweets=retweets,
                url=url,
                search_id=None,
                ranking=ranking
            ))

    print("click in id={}".format(clicked_doc_id))

    # store data in statistics table 1
    if clicked_doc_id in analytics_data.fact_clicks.keys():
        analytics_data.fact_clicks[clicked_doc_id] += 1
    else:
        analytics_data.fact_clicks[clicked_doc_id] = 1

    print("fact_clicks count for id={} is {}".format(clicked_doc_id, analytics_data.fact_clicks[clicked_doc_id]))

    return render_template('doc_details.html', document=document)

docs_stats = {}

@app.route('/stats', methods=['GET'])
def stats():
    """
    Show simple statistics example. ### Replace with dashboard ###
    :return:
    """

    search_id = request.args["search_id"]
    # Retrieve the initiation time of the search from the session

    for doc_id in analytics_data.fact_clicks.keys():
        row = corpus[int(doc_id)]

        # Recuperar la fecha y hora de inicio de la búsqueda desde la sesión
        search_initiation_time = session.get('search_initiation_time', datetime.now(pytz.utc))
        search_initiation_time = search_initiation_time.replace(tzinfo=pytz.utc)

        # Calculate time difference for each document inside the loop
        time_difference = datetime.now(pytz.utc) - search_initiation_time

        rel_query = session['last_search_query']

        count = analytics_data.fact_clicks[doc_id]

        ip_address = request.remote_addr

        user_agent = request.headers.get('User-Agent')
        os_info = httpagentparser.detect(user_agent)

        browser = session.get('user_agent')

        # Create a unique key for each document/query pair
        key = (doc_id, search_id)

        # Check if the key already exists in the dictionary
        if key in docs_stats:
            # Update the existing entry with the new information
            docs_stats[key].update(count)
        else:
            # Create a new entry in the dictionary
            docs_stats[key] = StatsDocument(
                id=row.id, 
                title=row.title, 
                tweet=row.tweet, 
                username=row.username, 
                date=row.date, 
                url=row.url,
                count=count, 
                time_difference=time_difference, 
                rel_query=rel_query, 
                search_initiation_time=search_initiation_time,
                ip_address=ip_address, 
                os_info=os_info, 
                browser=browser
            )

    # Convert the dictionary values to a list for rendering in the template
    clicks_data = list(docs_stats.values())

    # simulate sort by ranking
    clicks_data.sort(key=lambda doc: doc.count, reverse=True)

    return render_template('stats.html', clicks_data=clicks_data)


@app.route('/dashboard', methods=['GET'])
def dashboard():
    visited_docs = []
    print(analytics_data.fact_clicks.keys())
    for doc_id in analytics_data.fact_clicks.keys():
        d: Document = corpus[int(doc_id)]
        doc = ClickedDoc(doc_id, d.tweet, analytics_data.fact_clicks[doc_id])
        visited_docs.append(doc)

    # simulate sort by ranking
    visited_docs.sort(key=lambda doc: doc.counter, reverse=True)

    for doc in visited_docs: print(doc)
    return render_template('dashboard.html', visited_docs=visited_docs)

@app.route('/sentiment')
def sentiment_form():
    return render_template('sentiment.html')

@app.route('/sentiment', methods=['POST'])
def sentiment_form_post():
    text = request.form['text']
    nltk.download('vader_lexicon')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    score = ((sid.polarity_scores(str(text)))['compound'])
    return render_template('sentiment.html', score=score)

if __name__ == "__main__":
    app.run(port=8088, host="0.0.0.0", threaded=False, debug=True)

'''
import os
import pickle
from json import JSONEncoder
import httpagentparser
import nltk
from flask import Flask, render_template, request, session
from datetime import datetime, timedelta
import pytz

from myapp.analytics.analytics import AnalyticsData, ClickedDoc
from myapp.search.load_corpus import load_corpus
from myapp.search.objects import Document, ResultItem, StatsDocument
from myapp.search.search_engine import SearchEngine
from myapp.search.algorithms import create_index_tfidf

# *** Para usar el método `to_json` en objetos ***
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)

_default.default = JSONEncoder().default
JSONEncoder.default = _default

# Instancia de la aplicación Flask
app = Flask(__name__)

# Configuración de la clave secreta para la sesión
app.secret_key = 'afgsreg86sr897b6st8b76va8er76fcs6g8d7'
app.config['SESSION_COOKIE_NAME'] = 'IRWA_SEARCH_ENGINE'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)  # Caducidad de la sesión

# Función para guardar el índice en un archivo usando `pickle`
def save_index(index, tf, df, idf, filename="index.pkl"):
    with open(filename, "wb") as f:
        pickle.dump((index, tf, df, idf), f)
    print(f"Índice guardado en {filename}")

# Función para cargar el índice desde un archivo usando `pickle`
def load_index(filename="index.pkl"):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    print(f"No se encontró el archivo de índice en {filename}. Se va a calcular nuevamente...")
    return None

# Ruta completa del archivo de corpus
path = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(path, "farmers-protest-tweets.json")

# Cargar corpus desde el archivo `.pkl` si existe
if os.path.exists("corpus.pkl"):
    print("Cargando corpus desde archivo guardado...")
    with open("corpus.pkl", "rb") as pkl_file:
        corpus = pickle.load(pkl_file)
else:
    print("Generando nuevo corpus...")
    corpus = load_corpus(file_path)
    with open("corpus.pkl", "wb") as pkl_file:
        pickle.dump(corpus, pkl_file)
    print("Corpus guardado en `corpus.pkl`")

# Instanciar el motor de búsqueda
search_engine = SearchEngine()

# Instanciar los datos de análisis en memoria
analytics_data = AnalyticsData()

# Cargar o crear el índice TF-IDF
index_data = load_index()
if index_data is None:
    print("Creando nuevo índice...")
    index, tf, df, idf = create_index_tfidf(corpus)
    save_index(index, tf, df, idf)  # Guardar el nuevo índice
else:
    index, tf, df, idf = index_data

# Ruta principal "/"
@app.route('/')
def index():
    print("Iniciando ruta principal /...")

    session['some_var'] = "IRWA 2024 home"

    # Obtener datos del usuario (agente de usuario, IP)
    user_agent = request.headers.get('User-Agent')
    user_ip = request.remote_addr
    agent = httpagentparser.detect(user_agent)

    print(f"IP Remota: {user_ip} - JSON del navegador: {agent}")
    print(session)

    return render_template('index.html', page_title="Welcome")

# Ruta de búsqueda "/search"
@app.route('/search', methods=['POST'])
def search_form_post():
    search_query = request.form['search-query']

    session['last_search_query'] = search_query
    session['search_initiation_time'] = datetime.now(pytz.utc)

    search_id = analytics_data.save_query_terms(search_query)

    results = search_engine.search(search_query, search_id, corpus)

    found_count = len(results)
    session['last_found_count'] = found_count

    print(session)

    return render_template('results.html', results_list=results, page_title="Results", found_counter=found_count)

# Ruta de detalles del documento "/doc_details"
@app.route('/doc_details', methods=['GET'])
def doc_details():
    print("Detalles del documento en la sesión:")
    print(session)

    # Obtener parámetros de la consulta
    clicked_doc_id = request.args["id"]
    tweet = request.args["tweet"]
    username = request.args["username"]
    date = request.args["date"]
    likes = request.args["likes"]
    retweets = request.args["retweets"]
    url = request.args["url"]
    ranking = round(float(request.args["ranking"]), 4)

    document = ResultItem(
        id=clicked_doc_id,
        title=None,
        tweet=tweet,
        username=username,
        date=date,
        likes=likes,
        retweets=retweets,
        url=url,
        search_id=None,
        ranking=ranking
    )

    print(f"Click en el documento id={clicked_doc_id}")

    # Guardar datos en la tabla de clics de estadísticas
    analytics_data.increment_click(clicked_doc_id)

    print(f"Cantidad de clics para el id {clicked_doc_id}: {analytics_data.fact_clicks[clicked_doc_id]}")

    return render_template('doc_details.html', document=document)

# Ruta de estadísticas "/stats"
@app.route('/stats', methods=['GET'])
def stats():
    search_id = request.args["search_id"]
    docs_stats = {}

    for doc_id in analytics_data.fact_clicks.keys():
        row = corpus[int(doc_id)]

        # Recuperar la fecha de inicio de la búsqueda desde la sesión
        search_initiation_time = session.get('search_initiation_time', datetime.now(pytz.utc))
        search_initiation_time = search_initiation_time.replace(tzinfo=pytz.utc)

        # Calcular diferencia de tiempo
        time_difference = datetime.now(pytz.utc) - search_initiation_time

        rel_query = session['last_search_query']
        count = analytics_data.fact_clicks[doc_id]

        ip_address = request.remote_addr
        user_agent = request.headers.get('User-Agent')
        os_info = httpagentparser.detect(user_agent)

        browser = session.get('user_agent')

        # Crear una clave única para cada documento/búsqueda
        key = (doc_id, search_id)

        if key in docs_stats:
            docs_stats[key].update(count)
        else:
            docs_stats[key] = StatsDocument(
                id=row.id,
                title=row.title,
                tweet=row.tweet,
                username=row.username,
                date=row.date,
                url=row.url,
                count=count,
                time_difference=time_difference,
                rel_query=rel_query,
                search_initiation_time=search_initiation_time,
                ip_address=ip_address,
                os_info=os_info,
                browser=browser
            )

    clicks_data = list(docs_stats.values())
    clicks_data.sort(key=lambda doc: doc.count, reverse=True)

    return render_template('stats.html', clicks_data=clicks_data)

# Ruta del panel de control "/dashboard"
@app.route('/dashboard', methods=['GET'])
def dashboard():
    visited_docs = []
    print(analytics_data.fact_clicks.keys())
    
    # Convertir cada ClickedDoc a JSON antes de pasarla a la plantilla
    for doc_id in analytics_data.fact_clicks.keys():
        d: Document = corpus[int(doc_id)]
        doc = ClickedDoc(doc_id, d.tweet, analytics_data.fact_clicks[doc_id])
        
        # Convertir el objeto ClickedDoc a JSON
        visited_docs.append(doc.to_json())

    # Simular el orden por ranking
    visited_docs.sort(key=lambda doc: doc['counter'], reverse=True)

    # Aquí estamos pasando una lista de diccionarios (representaciones JSON de los ClickedDoc)
    return render_template('dashboard.html', visited_docs=visited_docs)


# Ruta para la página de análisis de sentimiento "/sentiment"
@app.route('/sentiment')
def sentiment_form():
    return render_template('sentiment.html')

# Ruta para procesar el análisis de sentimiento
@app.route('/sentiment', methods=['POST'])
def sentiment_form_post():
    text = request.form['text']
    nltk.download('vader_lexicon')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    score = sid.polarity_scores(str(text))['compound']
    return render_template('sentiment.html', score=score)

# Ejecutar la aplicación Flask
if __name__ == "__main__":
    app.run(port=8088, host="0.0.0.0", threaded=False, debug=True)


