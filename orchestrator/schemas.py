from marshmallow import Schema, fields

class OrchestratorSchema(Schema):
    question = fields.Str(required=True)
    response = fields.Str(dump_only=True)
