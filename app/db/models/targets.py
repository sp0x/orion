from db.models.bigintkey import *
from db.models.netlyt import *
from db.models.integration import *

#
# class TargetTimeConstraint(NetlytModel):
#     Id = BigIntPrimaryKey(primary_key=True)
#     Hours = pw.SmallIntegerField()
#     Days = pw.SmallIntegerField()
#     Seconds = pw.SmallIntegerField()
#
#     class Meta:
#         table_name = "TimeConstraint"
#
#
# class ModelTargets(NetlytModel):
#     Id = BigIntPrimaryKey(primary_key=True)
#     ModelId = pw.IntegerField()
#
#     class Meta:
#         table_name = "ModelTargets"
#
#
# class TargetConstraint(NetlytModel):
#     Id = BigIntPrimaryKey(primary_key=True)
#     # Model = pw.ForeignKeyField(Model, backref='ModelTargets', field='ModelTargetsId')
#     Type = pw.SmallIntegerField()
#     Key = pw.TextField()
#     After = pw.ForeignKeyField(TargetTimeConstraint, field='Id')
#     Before = pw.ForeignKeyField(TargetTimeConstraint, field='Id')
#     ModelTargetsId = pw.IntegerField()
#
#     class Meta:
#         table_name = "TargetConstraint"
