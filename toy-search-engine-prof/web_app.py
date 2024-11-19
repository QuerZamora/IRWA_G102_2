from flask import Flask
from flask import Flask, render_template, session
from flask import request
import httpagentparser  # for getting the user agent as json
from myapp.search.search_engine import SearchEngine
import os
from myapp.search.load_corpus import load_corpus
from json import JSONEncoder
import uuid

app = Flask(__name__)

#@app.route("/")
#def hello_world():
#    return "<p>Hello, World!</p>"



# *** for using method to_json in objects ***
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)


_default.default = JSONEncoder().default
JSONEncoder.default = _default

app.secret_key = 'afgsreg86sr897b6st8b76va8er76fcs6g8d7'
# open browser dev tool to see the cookies
app.session_cookie_name = 'IRWA_SEARCH_ENGINE'


#load corpus------------------------------------------
full_path = os.path.realpath(__file__) #get current path
path, filename = os.path.split(full_path)
# print(path + ' --> ' + filename + "\n")
# load documents corpus into memory.
file_path = path + "/farmers-protest-tweets.json"

# file_path = "../../tweets-data-who.json"
print(file_path)
corpus = load_corpus(file_path)
#print(corpus)
print("loaded corpus. first elem:", list(corpus.values())[0])


# instantiate our search engine
search_engine = SearchEngine()



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

    search_id = str(uuid.uuid4()) #I just get randomly an ID - unique identifier

    results = search_engine.search(search_query, search_id, corpus)

    found_count = len(results)
    session['last_found_count'] = found_count

    print(session)

    return render_template('results.html', results_list=results, page_title="Results", found_counter=found_count)






if __name__ == "__main__":
    app.run(port=8088, host="0.0.0.0", threaded=False, debug=True)
