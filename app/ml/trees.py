from Queue import Queue
from threading import Thread, Lock
from time import sleep

from ml.feature_extraction import BTree
from utils import parse_timespan


class BuildWorker(Thread):
    def __init__(self, par):
        super(BuildWorker,self).__init__()
        self.daemon = True
        self.par = par
        self.keep_working = True
        self.is_working = True

    def interrupt(self):
        self.keep_working = False

    def run(self):
        while self.par.work_available() and self.keep_working:
            job = self.par.__get_work__()
            userTree = BTree()
            # itemCount = 0
            uuid = job["_id"]
            # day_count = job["day_count"]
            days = job["daily_sessions"]

            day_index = 0
            for day in days:
                day_index += 1
                #print "adding sessions for {0} for day {1}/{2}".format(uuid, day_index, day_count)
                sessions = day
                day_items = []
                for session in sessions:
                    host = session["Domain"]
                    duration = parse_timespan(session["Duration"]).total_seconds()
                    day_items.append({'time': duration, 'label': host})
                userTree.build(day_items)
            #print "added daily sessions for {0}".format(uuid)
            self.par.push_result(userTree, uuid)
        print ("Worker stopping because no work is available")
        self.is_working = False


class MassTreeBuilder:
    def __init__(self, batch_size, store, filter, user_id_key):
        import settings
        self.userSessionTypeId = "598f20d002d2516dd0dbcee2"
        appId = "123123123"
        # sessionsPath = "testData/Netinfo/payingBrowsingSessionsDaySorted.csv"
        db = app.settings.get_db()
        self.documents_col = db.IntegratedDocument
        self.work_queue = Queue()
        self.lock = Lock()
        self.batch_size = batch_size
        self.res_queue = Queue()
        # self.remaining = self.documents_col.find({
        #     "TypeId": self.userSessionTypeId,
        #     "Document.is_paying": 0,
        #     # "Document.Created" : { "$lt" : week4Start }
        # }).distinct("Document.UserId")
        self.query_filter = filter
        self.user_id_key = user_id_key
        self.remaining = self.documents_col.find(self.query_filter).distinct(user_id_key)
        self.workers = []
        self.collecting_data = False
        self.io_lock = Lock()
        self.store = store
        [self.__fetch__() for _ in range(2)]

    def __fetch__(self):
        self.collecting_data = True
        with self.lock:
            ids = self.remaining[:self.batch_size]
            del self.remaining[:self.batch_size]
        #job items are users, and all their daily sessions
        match_filter = self.query_filter
        match_filter[self.user_id_key] = {"$in": ids}
        pipeline = [
            {"$match": match_filter },
            {"$group": {"_id": "$" + self.user_id_key,
                        "day_count": {"$sum": 1},
                        "daily_sessions": {"$push": "$Document.Sessions"}
                        }
             }
        ]
        user_groups = self.documents_col.aggregate(pipeline)
        with self.lock:
            for d in user_groups:
                self.work_queue.put_nowait(d)
            self.collecting_data = False

    def interrupt(self):
        for w in self.workers:
            w.interrupt()

    def __get_work__(self):
        with self.lock:
            rem = len(self.remaining)
        if self.work_queue.qsize() <= self.batch_size and rem > 0:
            if not self.collecting_data:
                t = Thread(target=self.__fetch__)
                t.daemon = True
                t.start()
        job = self.work_queue.get()
        self.work_queue.task_done()
        return job

    def is_working(self):
        return any(w.is_working for w in self.workers)

    def work_available(self):
        with self.lock:
            rem = len(self.remaining)
            queue_rem = len(self.work_queue.queue)
        return rem != 0 or queue_rem != 0

    def push_result(self, res, id=None):
        if self.store:
            from utils import abs_path, save
            import os
            path = abs_path(os.path.join("netinfo", id + ".pickle"))
            with self.io_lock:
                save(res, path)
        else:
            self.res_queue.put({'uuid': id,  'result': res})

    def build(self, max_threads=8):
        for i in xrange(max_threads):
            w = BuildWorker(self)
            w.start()
            self.workers.append(w)

    def make(self, max_threads=8):
        self.build(max_threads=max_threads)
        try:
            while self.work_available() and self.is_working():
                sleep(1)
        except KeyboardInterrupt:
            self.interrupt()
        results = self.get_result()
        return results

    def get_result(self):
        if not self.work_available():
            return list(self.res_queue.queue)
        return []