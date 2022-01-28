"""
Fake implementation of faster-fifo that just routes all function calls to multiprocessing.Queue.
Can be useful on platforms where faster-fifo does not work, e.g. Windows.
"""

import multiprocessing
from queue import Empty, Full


class Queue:
    def __init__(self, max_size_bytes=200000):
        self.q = multiprocessing.Queue(max_size_bytes)

    def close(self):
        self.q.close()

    def is_closed(self):
        """Not implemented."""
        return False

    def put(self, x, block=True, timeout=float(1e3)):
        self.q.put(x, block, timeout)

    def put_nowait(self, x):
        return self.put(x, block=False)

    def get_many(self, block=True, timeout=float(1e3), max_messages_to_get=int(1e9)):
        msgs = []

        while len(msgs) < max_messages_to_get:
            try:
                if len(msgs) == 0:
                    msg = self.q.get(block, timeout)
                else:
                    msg = self.q.get_nowait()

                msgs.append(msg)
            except Empty:
                break

        return msgs

    def get_many_nowait(self, max_messages_to_get=int(1e9)):
        return self.get_many(block=False, max_messages_to_get=max_messages_to_get)

    def get(self, block=True, timeout=float(1e3)):
        return self.get_many(block=block, timeout=timeout, max_messages_to_get=1)[0]

    def get_nowait(self):
        return self.get(block=False)

    def qsize(self):
        return self.q.qsize()

    def empty(self):
        return self.q.empty()

    def full(self):
        return self.q.full()

    def join_thread(self):
        self.q.join_thread()

    def cancel_join_thread(self):
        self.q.cancel_join_thread()

    # def put_many(self, xs, block=True, timeout=DEFAULT_TIMEOUT):
    #     assert isinstance(xs, (list, tuple))
    #     xs = [_ForkingPickler.dumps(ele).tobytes() for ele in xs]
    #
    #     _len = len
    #     msgs_buf = (c_size_t * _len(xs))()
    #     size_buf = (c_size_t * _len(xs))()
    #
    #     for i, ele in enumerate(xs):
    #         msgs_buf[i] = bytes_to_ptr(ele)
    #         size_buf[i] = _len(ele)
    #
    #     # explicitly convert all function parameters to corresponding C-types
    #     cdef
    #     void * c_q_addr = < void * > q_addr(self)
    #     cdef
    #     void * c_buf_addr = < void * > buf_addr(self)
    #
    #     cdef
    #     const
    #     void ** c_msgs_buf_addr = < const
    #     void ** > caddr(msgs_buf)
    #     cdef
    #     const
    #     size_t * c_size_buff_addr = < const
    #     size_t * > caddr(size_buf)
    #
    #     cdef
    #     size_t
    #     c_len_x = _len(xs)
    #     cdef
    #     int
    #     c_block = block
    #     cdef
    #     float
    #     c_timeout = timeout
    #
    #     cdef
    #     int
    #     c_status = 0
    #
    #     with nogil:
    #         c_status = Q.queue_put(
    #             c_q_addr, c_buf_addr, c_msgs_buf_addr, c_size_buff_addr, c_len_x,
    #             c_block, c_timeout,
    #         )
    #
    #     status = c_status
    #
    #     if status == Q.Q_SUCCESS:
    #         pass
    #     elif status == Q.Q_FULL:
    #         raise Full()
    #     else:
    #         raise Exception(f'Unexpected queue error {status}')

    def put_many(self, x, block=True, timeout=float(1e3)):
        assert isinstance(x, (list, tuple))
        for message in x:
            self.q.put(message, block, timeout)

    def put_many_nowait(self, x, timeout=float(1e3)):
        return self.put_many(x, block=False, timeout=timeout)
