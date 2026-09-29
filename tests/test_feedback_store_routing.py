import copy
import unittest
from types import SimpleNamespace
from maple_guard.memory_backend import MemoryBackendBundle
from maple_guard import maple_guard_core as ep

class Store:
    """Minimal persistent-store contract: IDs are local to each recipient store."""
    def __init__(self,name):
        self.name=name
        self.items={}
        self.updates=[]
    def add_memory(self,*,metadata,**kwargs):
        key=self.name+":"+metadata["maple_guard_id"]
        self.items[key]=copy.deepcopy(metadata)
        return key
    def update_value(self,key,reward):
        if key not in self.items:raise ValueError("memory absent from recipient store")
        self.items[key]["q_value"]=reward
        self.updates.append(key)
        return reward

class FeedbackStoreRouting(unittest.TestCase):
    def setUp(self):
        self.args=SimpleNamespace(agents=2,strict_comparison=True,memory_run_id="routing-regression")
        self.bundle=MemoryBackendBundle(self.args,ep.MemoryEntry)
        for backend in self.all_backends():
            backend._service=Store(backend.name)
    def all_backends(self):
        return [*self.bundle.private_backends.values(),self.bundle.shared_backend,self.bundle.quarantine_backend]
    def entry(self):
        task=ep.TaskExample("task","Question",[("A","yes"),("B","no")],"A","B",raw={})
        return ep.create_benign_memory(task,1,"A",True,"test")
    def test_recipient_private_store_receives_feedback_not_author_store(self):
        entry=self.entry()
        self.bundle.add_private(0,entry)
        try:self.bundle.update_value(entry,False)
        except RuntimeError as exc:self.fail(str(exc))
        self.assertEqual(self.bundle.private_backends[0]._service.updates,["agent_0:"+entry.memory_id])
        self.assertEqual(self.bundle.private_backends[1]._service.updates,[])
        self.assertEqual(entry.origin_agent,1)
        self.assertEqual(entry.utility_q,-1.0)
    def test_rehydrated_candidate_keeps_physical_store_routing(self):
        entry=self.entry()
        self.bundle.add_private(0,entry)
        backend=self.bundle.private_backends[0]
        key=backend.backend_ids[entry.memory_id]
        metadata=backend._service.items[key]
        backend.entries.clear()
        retrieved=backend._candidate_to_entry({"memory_id":key,"metadata":metadata,"q_estimate":0.0})
        try:self.bundle.update_value(retrieved,True)
        except RuntimeError as exc:self.fail(str(exc))
        self.assertEqual(backend._service.updates,[key])
        self.assertEqual(retrieved.origin_agent,1)
    def test_shared_store_feedback_is_not_sent_to_private_author(self):
        entry=self.entry();entry.memory_scope="team"
        self.bundle.add_shared(entry)
        self.bundle.update_value(entry,True)
        self.assertEqual(self.bundle.shared_backend._service.updates,["shared:"+entry.memory_id])
        self.assertEqual(self.bundle.private_backends[1]._service.updates,[])
    def test_unregistered_entry_cannot_silently_skip_strict_feedback(self):
        with self.assertRaisesRegex(RuntimeError,"store|registered"):
            self.bundle.update_value(self.entry(),True)

if __name__=="__main__":unittest.main()
