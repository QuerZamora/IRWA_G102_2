import json
import random
from datetime import datetime


import json
import random
from datetime import datetime

class AnalyticsData:
    """
    An in-memory persistence object to track and store analytics data.
    """

    def __init__(self):
        # Ahora fact_clicks será una variable de instancia
        self.fact_clicks = dict([])

    def save_query_terms(self, terms: str) -> int:
        """
        Save the search query terms, returning a random search_id.
        """
        print(self)
        return random.randint(0, 100000)

    def increment_click(self, doc_id: str):
        """
        Increment the click count for a specific document ID.
        """
        if doc_id in self.fact_clicks:
            self.fact_clicks[doc_id] += 1
        else:
            self.fact_clicks[doc_id] = 1
        print(f"Incremented click count for doc_id {doc_id}. New count: {self.fact_clicks[doc_id]}")



class ClickedDoc:
    def __init__(self, doc_id, description, counter):
        self.doc_id = doc_id
        self.description = description
        self.counter = counter
        self.time_difference = None
        self.rel_query = None

    def to_json(self):
        return self.__dict__

    def __str__(self):
        """
        Print the object content as a JSON string
        """
        return json.dumps(self)