from marshmallow import Schema, fields


class AssistantSchema(Schema):
    question = fields.Str(required=True)
    response = fields.Str(dump_only=True)
