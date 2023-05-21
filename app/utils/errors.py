from flask import jsonify

class InvalidUsage(Exception):
    """ Custom exception class for handling invalid usage in API calls. """
    
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code
    
    def to_response(self):
        response = jsonify({
            'message': str(self),
            'status_code': self.status_code
        })
        response.status_code = self.status_code
        return response
