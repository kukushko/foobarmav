#!/usr/bin/python

import time
import threading
import select
import thread
from socket import *

_DEBUG = False

class _Connection:

    def __init__(self, conn_id, cs, ca, target_addr, buffer_size):
        self.conn_id = conn_id
        self.buffer_size = buffer_size
        self.start_time = time.time()
        self.cs = cs
        self.ca = ca
        self.client_bytes_received = 0L
        self.target_bytes_received = 0L
        self.client_bytes_sent = 0L
        self.target_bytes_sent = 0L
        self.target_addr = target_addr
        self.ts = socket(AF_INET, SOCK_STREAM)
        self.ts.setblocking(False)
        self.ts.connect_ex(target_addr)
        self.target_connected = False
        self.cs_closed = False
        self.ts_closed = False
        self.cs_read_complete = False
        self.ts_read_complete = False
        self.from_ts_buffers = []
        self.from_cs_buffers = []
        self.is_closed = False
    
    def select_cs_rd(self):
        return not self.cs_read_complete
    
    def select_ts_rd(self):
        return not self.ts_read_complete

    def select_cs_wr(self):
        return len(self.from_ts_buffers) > 0 and not self.cs_closed
    
    def select_ts_wr(self):
        return (not self.target_connected or len(self.from_cs_buffers) > 0) and not self.ts_closed
    
    def _log(self, msg):
        print("%s:%s - %s" % (self.ca[0], self.ca[1], msg))
    
    def close(self):
        if not self.is_closed:
            et = time.time()
            duration = et - self.start_time
            self._log("closing conn %s, client rcv/snd: %s/%s, target rcv/snd: %s/%s, duration: %.3fs" % (self.conn_id, self.client_bytes_received, self.client_bytes_sent, self.target_bytes_received, self.target_bytes_sent, duration))
            self.cs.close()
            self.ts.close()
            self.is_closed = True
    
    def read_ts(self):
        data = self.ts.recv(self.buffer_size)
        if not data:
            if _DEBUG:
                self._log("enqueue connection close to client")
            self.from_ts_buffers.append(None)
            self.ts_read_complete = True
        else:
            if _DEBUG:
                self._log("enqueue %s bytes to client" % len(data))
            self.from_ts_buffers.append(data)
            self.target_bytes_received += len(data)
    
    def read_cs(self):
        data = self.cs.recv(self.buffer_size)
        if not data:
            if _DEBUG:
                self._log("enqueue connection close to target")
            self.from_cs_buffers.append(None)
            self.cs_read_complete = True
        else:
            if _DEBUG:
                self._log("enqueue %s bytes to target" % len(data))
            self.from_cs_buffers.append(data)
            self.client_bytes_received += len(data)
    
    def write_ts(self):
        if not self.target_connected:
            if _DEBUG:
                self._log("target connected")
            self.target_connected = True
        else:
            data_to_send = self.from_cs_buffers[0]
            if data_to_send is None:
                self.from_cs_buffers = []
                if _DEBUG:
                    self._log("shutdown target")
                try:
                    self.ts.shutdown(SHUT_WR)
                except Exception, x:
                    if _DEBUG:
                        self._log("target shutdown failed - %s" % x)
                    self.cs.close()
                    self.cs_closed = True
                self.ts_closed = True
            else:
                #self._log(data_to_send)
                sent = self.ts.send(data_to_send)
                if _DEBUG:
                    self._log("%s bytes of %s sent to target" % (sent, len(data_to_send)))
                self.target_bytes_sent += sent
                if sent == len(data_to_send):
                    self.from_cs_buffers.remove(data_to_send)
                else:
                    self.from_cs_buffers[0] = buffer(data_to_send, sent)
    
    def write_cs(self):
        data_to_send = self.from_ts_buffers[0]
        if data_to_send is None:
            self.from_ts_buffers = []
            if _DEBUG:
                self._log("shutdown client")
            try:
                self.cs.shutdown(SHUT_WR)
            except Exception, x:
                if _DEBUG:
                    self._log("client shutdown failed - %s" % x)
                self.ts.close()
                self.ts_closed = True
            self.cs_closed = True
        else:
            #self._log(data_to_send)
            sent = self.cs.send(data_to_send)
            if _DEBUG:
                self._log("%s bytes of %s sent to target" % (sent, len(data_to_send)))
            self.client_bytes_sent += sent
            if sent == len(data_to_send):
                self.from_ts_buffers.remove(data_to_send)
            else:
                self.from_ts_buffers[0] = buffer(data_to_send, sent)

class Proxy:

    def __init__(self, listen_addr, target_addr, backlog = 5, num_threads = 10, buffer_size = 4096):
        self.__listen_addr = listen_addr
        self.__num_threads = num_threads
        self.__target_addr = target_addr
        self.__backlog = backlog
        self.__stop_pending = False
        self.__buffer_size = buffer_size
        self.__connections = []
        self.__connection_by_socket = {}
        self.__conn_lock = threading.Lock()
        self.__connection_counter = 0L
        self.__finish_event = threading.Event()
    
    def stop(self):
        self.__stop_pending = True
        self.__finish_event.wait()
    
    def _log(self, msg):
        print("%s:%s - %s" % (self.__listen_addr[0], self.__listen_addr[1], msg))
    
    def __register_conn(self, conn):
        self.__conn_lock.acquire()
        self.__connections.append(conn)
        self.__connection_by_socket[conn.cs] = conn
        self.__connection_by_socket[conn.ts] = conn
        self.__conn_lock.release()
    
    def __pick_rdset(self, thread_id):
        self.__conn_lock.acquire()
        cs_rd_set = [c.cs for c in filter(lambda x: x.select_cs_rd(), self.__relevant_conns(thread_id))] 
        ts_rd_set = [c.ts for c in filter(lambda x: x.select_ts_rd(), self.__relevant_conns(thread_id))]
        self.__conn_lock.release()
        return cs_rd_set + ts_rd_set
    
    def __pick_wrset(self, thread_id):
        self.__conn_lock.acquire()
        cs_wr_set = [c.cs for c in filter(lambda x: x.select_cs_wr(), self.__relevant_conns(thread_id))]
        ts_wr_set = [c.ts for c in filter(lambda x: x.select_ts_wr(), self.__relevant_conns(thread_id))]
        self.__conn_lock.release()
        return cs_wr_set + ts_wr_set
    
    def __pick_xset(self, thread_id):
        self.__conn_lock.acquire()
        result = [c.cs for c in self.__relevant_conns(thread_id)] + [c.ts for c in self.__relevant_conns(thread_id)]
        self.__conn_lock.release()
        return result
    
    def __close_by_socket(self, sl):
        self.__conn_lock.acquire()
        for s in sl:
            conn = self.__connection_by_socket.get(s, None)
            if conn:
                conn.close()
                del self.__connection_by_socket[conn.ts]
                del self.__connection_by_socket[conn.cs]
                self.__connections.remove(conn)
            else:
                self._log("cannot find connection for socket to close")
        self.__conn_lock.release()
    
    def __process_rd(self, sl):
        cs_read_conns = []
        ts_read_conns = []
        self.__conn_lock.acquire()
        for s in sl:
            conn = self.__connection_by_socket[s]
            if conn.ts == s:
                ts_read_conns.append(conn)
            else:
                cs_read_conns.append(conn)
        self.__conn_lock.release()
        for conn in cs_read_conns:
            conn.read_cs()
        for conn in ts_read_conns:
            conn.read_ts()
    
    def __process_wr(self, sl):
        cs_write_conns = []
        ts_write_conns = []
        self.__conn_lock.acquire()
        for s in sl:
            conn = self.__connection_by_socket[s]
            if conn.ts == s:
                ts_write_conns.append(conn)
            else:
                cs_write_conns.append(conn)
        self.__conn_lock.release()
        for conn in cs_write_conns:
            conn.write_cs()
        for conn in ts_write_conns:
            conn.write_ts()

    def __relevant_conns(self, thread_id):
        return filter(lambda x: (x.conn_id % self.__num_threads) == thread_id, self.__connections)
        
    def __remove_dead(self, thread_id):
        self.__conn_lock.acquire()
        for c in self.__relevant_conns(thread_id):
            if c.ts_closed and c.cs_closed:
                del self.__connection_by_socket[c.cs]
                del self.__connection_by_socket[c.ts]
                self.__connections.remove(c)
                c.close()
        self.__conn_lock.release()
    
    def start(self):
        thread.start_new_thread(self.__serve, ())
    
    def __serve(self):
        self.__stopped = False
        def connection_server(thread_id):
            while not self.__stopped:
                rds = self.__pick_rdset(thread_id)
                wrs = self.__pick_wrset(thread_id)
                xs = self.__pick_xset(thread_id)
                #self._log("rds: %s, wrs: %s, xs: %s" % (len(rds), len(wrs), len(xs)))
                r,w,x = select.select(rds, wrs, xs, 1.0)
                #self._log("r,w,x: %s,%s,%s" % (len(r), len(w), len(x)))
                self.__close_by_socket(x)
                self.__process_rd(r)
                self.__process_wr(w)
                self.__remove_dead(thread_id)
        
        for i in xrange(self.__num_threads):
            thread.start_new_thread(connection_server, (i, ))
        
        self.__stop_pending = False
        ss = socket(AF_INET, SOCK_STREAM)
        ss.bind(self.__listen_addr)
        ss.listen(self.__backlog)
        ss.setblocking(False)
        while not self.__stop_pending:
            r,w,x = select.select([ss],[],[ss], 1.0)
            if x:
                self._log("error on server socket, terminating")
                break
            if r:
                cs, ca = ss.accept()
                self.__connection_counter += 1
                self._log("accepted connection %s - %s:%s" % (self.__connection_counter, ca[0], ca[1]))
                conn = _Connection(self.__connection_counter, cs, ca, self.__target_addr, self.__buffer_size)
                self.__register_conn(conn)
                
        self._log("serve cycle stopped")
        self.__stopped = True
        ss.close()
        self.__finish_event.set()


p = Proxy(("127.0.0.1", 10001), ("python.org", 443))
p.start()
raw_input()
p.stop()

