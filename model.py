import logging
import time
import os
from djl_python import Input
from djl_python import Output
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np


class BartModel(object):
    """
    Deploying Bart with DJL Serving
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, properties: dict):
        """
        Initialize model.
        """
        print(os.listdir())
        logging.info("-----------------")
        logging.info(properties)
        
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
        model = AutoModel.from_pretrained("facebook/bart-large")
        
        self.model_name = properties.get("model_id")
        self.task = properties.get("task")
        logging.info("-----------------")
        logging.info(self.model_name)
        logging.info("-----------------")
        logging.info(self.task)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.initialized = True

    def inference(self, inputs):
        """
        Custom service entry point function.

        :param inputs: the Input object holds the text for the BART model to infer upon
        :return: the Output object to be send back
        """

        #sample input: "This is the sample text that I am passing in"
        try:
            data = inputs.get_as_string()
            logging.info("-----------------")
            logging.info(data)
            logging.info(type(data))
            logging.info("-----------------")
            inputs = self.tokenizer(data, return_tensors="pt")
            preds = self.model(**inputs)
            logging.info("-----------------")
            logging.info(type(preds))
            logging.info("-----------------")
            res = preds.last_hidden_state.detach().cpu().numpy().tolist() #convert to JSON Serializable object
            outputs = Output()
            outputs.add_as_json(res)
        except Exception as e:
            logging.exception("inference failed")
            # error handling
            outputs = Output().error(str(e))
        
        print(outputs)
        print(type(outputs))
        print("Returning inference---------")
        return outputs


_service = BartModel()


def handle(inputs: Input):
    """
    Default handler function
    """
    if not _service.initialized:
        # stateful model
        _service.initialize(inputs.get_properties())
    
    if inputs.is_empty():
        return None

    return _service.inference(inputs)
