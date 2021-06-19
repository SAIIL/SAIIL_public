#!/usr/bin/env python3
# Example reader/dataset code for SAGES protobuf example schema.
#
# Requires: python3, protobuf3, 
#
# To create the parser from the library:
#
# protoc --python_out=./ ./sages.proto
#
# To run:
#
# python ./sages_protobuf_example.py
#
import json
import time

from google.protobuf.json_format import MessageToDict, MessageToJson
from torch.utils.data import Dataset

from sages_pb2 import *

example_event = Event()
example_event2 = Event()
example_event3 = Event()
example_event4 = Event()

example_event.name = 'A'
example_event.type = 'phase'
example_event.temporal_span.start = time.time()
example_event.temporal_span.end = time.time() + 100
example_event.temporal_span.start_exact = True
example_event.temporal_span.end_exact = True

example_event2.name = 'B'
example_event2.type = 'action'
example_event2.temporal_span.start = time.time()
example_event2.temporal_span.end = time.time() + 100
example_event2.temporal_span.start_exact = True
example_event2.temporal_span.end_exact = True

example_event3.name = 'C'
example_event3.type = 'bleeding'
example_event3.temporal_span.start = time.time()
example_event3.temporal_span.end = time.time() + 100
example_event3.temporal_span.start_exact = True
example_event3.temporal_span.end_exact = True

example_event4.name = 'D'
example_event4.type = 'clipping_tool_bbox'
example_event4.temporal_span.start = time.time()
example_event4.temporal_span.end = time.time()
example_event4.temporal_span.start_exact = True
example_event4.temporal_span.end_exact = True
example_event4.spatial_span.elements.append(Coordinate2D(x=2,y=2))
example_event4.spatial_span.elements.append(Coordinate2D(x=1,y=2))
example_event4.spatial_span.elements.append(Coordinate2D(x=1,y=1))
example_event4.spatial_span.elements.append(Coordinate2D(x=2,y=1))
example_relation = Relation(
)
entity_id1='0001'
entity_id2='0002'
entity_id3='0003'
entity_id4='0004'

steps_list = Track(name='surgical_steps')
steps_list.entities.append(SurgeryEntity(event=example_event,entity_id=entity_id1))
steps_list.entities.append(SurgeryEntity(event=example_event2,entity_id=entity_id2))

event_list = Track(name='events')
event_list.entities.append(SurgeryEntity(event=example_event3,entity_id=entity_id3))

tool_list = Track(name='tool_annotations')
tool_list.entities.append(SurgeryEntity(event=example_event4,entity_id=entity_id4))

example_relation.type = 'causes'
example_relation.entity_ids.append(entity_id1)
example_relation.entity_ids.append(entity_id2)
example_relation.additional_information['name']='completely_specifies'
example_relation.additional_information['certainty']='0.5'

example_relation2 = Relation(
)

example_relation2.type = 'causes'
example_relation2.entity_ids.append(entity_id2)
example_relation2.entity_ids.append(entity_id3)
example_relation2.additional_information['name']='completely_specifies'

example_relation3 = Relation(
)
example_relation3.type = 'marks_beginning'
example_relation3.entity_ids.append(entity_id1)
example_relation3.entity_ids.append(entity_id2)


annotation_set=AnnotationSet()
annotation_set.tracks.tracks.append(steps_list)
annotation_set.tracks.tracks.append(event_list)
annotation_set.tracks.tracks.append(tool_list)
annotation_set.temporal_relations.append(example_relation)
annotation_set.temporal_relations.append(example_relation2)
annotation_set.annotator_id = '007'

# import IPython;IPython.embed()

with open('example.json', 'w') as fp:
    json.dump(MessageToDict(annotation_set), fp, indent=4)

with open('example.pb', 'wb') as fp:
    fp.write(example_relation.SerializeToString())
    fp.write(example_relation2.SerializeToString())

with open('example_events.pb', 'wb') as fp:
    fp.write(annotation_set.SerializeToString())

class ProtobufDataset(Dataset):
    def __init__(self,filename):
        with open(filename,'rb') as fp:
            self.annotation_set = AnnotationSet()
            self.annotation_set.ParseFromString(fp.read())
            self.all_entities=[]
            for tr  in self.annotation_set.tracks.tracks:
                for entity in tr.entities:
                    self.all_entities.append({'track_name':tr.name,'entity':entity})

    def __len__(self):
        return len(self.all_entities)

    def __getitem__(self, item):
        # TODO(guy.rosman): this wouldn't work with a dataloader as-is..
        return self.all_entities[item]['entity']

    def __iter__(self):
        def generator():
            for i in range(len(self.all_entities)):
                yield self.all_entities[i]
        return generator()

class ProtobufSpatialDataset(Dataset):
    def __init__(self,filename):
        with open(filename,'rb') as fp:
            self.annotation_set = AnnotationSet()
            self.annotation_set.ParseFromString(fp.read())
            self.all_entities=[]
            for tr  in self.annotation_set.tracks.tracks:
                for entity in tr.entities:
                    if len(entity.event.spatial_span.elements)>0:
                        self.all_entities.append({'track_name':tr.name,'entity':entity})

    def __len__(self):
        return len(self.all_entities)

    def __getitem__(self, item):
        # TODO(guy.rosman): this wouldn't work with a dataloader as-is..
        return self.all_entities[item]['entity']

    def __iter__(self):
        def generator():
            for i in range(len(self.all_entities)):
                yield self.all_entities[i]
        return generator()


proto_dataset=ProtobufDataset('example_events.pb')
proto_spatial_dataset=ProtobufSpatialDataset('example_events.pb')


print('Annotation_set: {}'.format(str(annotation_set)))

print('\n\nDataset has {} entities.'.format(len(proto_dataset)))
for itm in proto_dataset:
    print(itm)

print('\n\nDataset has {} spatial annotations.'.format(len(proto_spatial_dataset)))
for itm in proto_spatial_dataset:
    print(itm)

uid_number=10;
model_definition=AnnotationModelDefinition()
phase_names = ['phase_execution','phase_access']
step_names = ['step_obtain_CVS']
task_names = ['task_dissect']
action_verb_names=['dissect','clip']
phases_uid={}
steps_uid={}
tasks_uid={}
action_verb_uid={}

for pn in phase_names:
    uid_number += 1
    uid='{:06d}'.format(uid_number)
    phases_uid[pn]=uid
    model_definition.entity_types.append(SurgicalObject(object_name=pn, object_uid=uid))
for sn in step_names:
    uid_number += 1
    uid='{:06d}'.format(uid_number)
    steps_uid[sn]=uid
    model_definition.entity_types.append(SurgicalObject(object_name=sn, object_uid=uid))
for tn in task_names:
    uid_number += 1
    uid='{:06d}'.format(uid_number)
    tasks_uid[tn]=uid
    model_definition.entity_types.append(SurgicalObject(object_name=tn, object_uid=uid))
for an in action_verb_names:
    uid_number += 1
    uid='{:06d}'.format(uid_number)
    action_verb_uid[an]=uid
    model_definition.entity_types.append(SurgicalObject(object_name=an, object_uid=uid))
model_definition.version='0.1'

def create_relation(type,ids):
    relation = Relation(type=type)
    for id in ids:
        relation.entity_ids.append(id)
    return relation


model_definition.constraints.append(create_relation('includes',[phases_uid['phase_execution'],steps_uid['step_obtain_CVS']]))
model_definition.constraints.append(create_relation('includes',[steps_uid['step_obtain_CVS'],tasks_uid['task_dissect']]))
model_definition.constraints.append(create_relation('includes',[tasks_uid['task_dissect'],action_verb_uid['dissect']]))

import IPython;IPython.embed()
