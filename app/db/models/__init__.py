from db.models.bigintkey import *
from db.models.netlyt import *
from db.models.integration import *
from db.models.targets import *
#
#
# class Field(NetlytModel):
#     Id = BigIntPrimaryKey(primary_key=True)
#     Name = pw.TextField()
#     Type = pw.TextField()
#     ExtrasId = pw.IntegerField()
#     DataEncoding = pw.SmallIntegerField()
#     IntegrationId = pw.IntegerField()
#     ModelTargetsId = pw.IntegerField()
#
#     class Meta:
#         table_name = "Fields"

# TODO: Remove this by separating encoding/decoding from netlyt & orion
class FieldExtra(NetlytModel):
    Id = BigIntPrimaryKey(primary_key=True)
    Key = pw.TextField()
    Value = pw.TextField()
    FieldId = pw.IntegerField()
    Type = pw.IntegerField()
    FieldExtrasId = pw.IntegerField()

    class Meta:
        table_name = "FieldExtra"

# class ApiKey(NetlytModel):
#    Id = BigIntPrimaryKey(primary_key=True)
#    Endpoint = pw.CharField(max_length=1000)
#    AppId = pw.CharField(max_length=255)
#    AppSecret = pw.CharField(max_length=500)
#    Type = pw.CharField(max_length=50)#
#
#    class Meta:
#        table_name = "ApiKeys"

#
# class FeatureGenerationTasks(NetlytModel):
#     Id = BigIntPrimaryKey(primary_key=True)
#     Status = pw.IntegerField()
#     ModelId = pw.BigIntegerField()
#
#     class Meta:
#         table_name = "FeatureGenerationTasks"
#
#
# class ModelPerformance(NetlytModel):
#     Id = BigIntPrimaryKey(primary_key=True)
#     ModelId = pw.BigIntegerField()
#     TrainedTs = pw.DateTimeField()
#     Accuracy = pw.FloatField()
#     FeatureImportance = JSONField()
#     AdvancedReport = JSONField()
#     ReportUrl = pw.CharField(max_length=255)
#     TestResultsUrl = pw.CharField(max_length=255)
#     TargetName = pw.CharField(max_length=255)
#     WeeklyUsage = JSONField()
#     MonthlyUsage = JSONField()
#     LastRequestTs = pw.DateTimeField()
#     LastRequestIP = pw.CharField(max_length=255)
#
#     class Meta:
#         table_name = "ModelTrainingPerformance"
#
#
# class TrainingTask(NetlytModel):
#     Id = BigIntPrimaryKey(primary_key=True)
#     ModelId = pw.BigIntegerField()
#     Status = pw.IntegerField()
#
#     class Meta:
#         table_name = "TrainingTasks"
#
#
# def get_integration_schema(ign):
#     qr_fields = (Field
#                  .select(Field)
#                  .join(Integration, on=(Integration.Id == Field.IntegrationId))
#                  .where(Integration.Id == ign.Id))
#     return qr_fields
#
#
# def get_model_target_fields(mlmodel):
#     qr_integration = (Field
#                       .select(Field)
#                       .join(ModelTargets, on=(ModelTargets.Id == Field.ModelTargetsId))
#                       .join(Model, on=(Model.Id == ModelTargets.ModelId))
#                       .where(Model.Id == mlmodel.Id))
#     fields = qr_integration
#     return fields
