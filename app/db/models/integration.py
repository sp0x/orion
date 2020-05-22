from db.models.bigintkey import *
from db.models.netlyt import *
#
# class Integration(NetlytModel):
#     Id = BigIntPrimaryKey(primary_key=True)
#     OwnerId = pw.CharField(max_length=1000)
#     FeatureScript = pw.CharField(max_length=1000)
#     Name = pw.CharField(255)
#     DataEncoding = pw.SmallIntegerField()
#     APIKeyId = pw.IntegerField()
#     PublicKeyId = pw.IntegerField()
#     DataFormatType = pw.TextField()
#     Source = pw.TextField()
#     Collection = pw.TextField()
#     DataIndexColumn = pw.TextField()
#     DataTimestampColumn = pw.TextField()
#     FeaturesCollection = pw.TextField()
#     #Models = pw.ForeignKeyField(Model, backref='ModelIntegrations')
#
#     class Meta:
#         table_name = "Integrations"
