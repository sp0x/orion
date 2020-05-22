# distutils: language = c++

from libcpp.string cimport string

import random
import re

import six
from six.moves import zip, xrange

from .lang_detect_exception import ErrorCode, LangDetectException
from .language import Language
from .utils.ngram import NGram
from .utils.unicode_block import unicode_block
import numpy
cimport numpy as np

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = numpy.int

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int_t DTYPE_t


from libc.stdlib cimport malloc, free
cdef public int* global_q = <int*>malloc(sizeof(int) * 2)
global_q[0] = 10000
global_q[1] = 1000
cdef public double* global_f = <double*>malloc(sizeof(double) * 4)
global_f[0] = 0.1
global_f[1] = 0.99999
global_f[2] = 0.5
global_f[3] = 0.05

def call_at_end():
    free(global_q)
    free(global_f)

cdef extern from "std.h":
    double sum(double *arr, size_t siz)
#
# def cStdDev(ndarray[np.float64_t, ndim=1] a not None):
#     return std_dev(<double*> a.data, a.size)

cdef class Detector:
    '''
    Detector class is to detect language from specified text.
    Its instance is able to be constructed via the factory class DetectorFactory.

    After appending a target text to the Detector instance with .append(string),
    the detector provides the language detection results for target text via .detect() or .get_probabilities().

    .detect() method returns a single language name which has the highest probability.
    .get_probabilities() methods returns a list of multiple languages and their probabilities.

    The detector has some parameters for language detection.
    See set_alpha(double), .set_max_text_length(int) .set_prior_map(dict).

    Example:

        from langdetect.detector_factory import DetectorFactory
        factory = DetectorFactory()
        factory.load_profile('/path/to/profile/directory')

        def detect(text):
            detector = factory.create()
            detector.append(text)
            return detector.detect()

        def detect_langs(text):
            detector = factory.create()
            detector.append(text)
            return detector.get_probabilities()
    '''

    cdef double ALPHA_DEFAULT
    cdef double ALPHA_WIDTH

    cdef int ITERATION_LIMIT
    cdef double PROB_THRESHOLD
    cdef double CONV_THRESHOLD
    cpdef int BASE_FREQ
    UNKNOWN_LANG = 'unknown'

    URL_RE = re.compile(r'https?://[-_.?&~;+=/#0-9A-Za-z]{1,2076}')
    MAIL_RE = re.compile(r'[-_.0-9A-Za-z]{1,64}@[-_0-9A-Za-z]{1,255}[-_.0-9A-Za-z]{1,255}')

    cdef public string text
    cdef public object random
    cdef double alpha
    cdef double seed
    cdef int max_text_length
    cdef int n_trial
    cdef list langlist
    cdef dict word_lang_prob_map
    cdef np.ndarray langprob
    cdef list prior_map
    cdef int verbose

    def __init__(self, factory):
        self.BASE_FREQ = global_q[0]
        self.ITERATION_LIMIT = global_q[1]

        self.PROB_THRESHOLD = global_f[0]
        self.CONV_THRESHOLD = global_f[1]
        self.ALPHA_DEFAULT = global_f[2]
        self.ALPHA_WIDTH = global_f[3]

        self.word_lang_prob_map = factory.word_lang_prob_map
        self.langlist = factory.langlist
        self.seed = factory.seed
        self.random = random.Random()
        self.text = ''
        self.langprob = None

        self.alpha = self.ALPHA_DEFAULT
        self.n_trial = 7
        self.max_text_length = 10000
        self.prior_map = None
        self.verbose = False

    def reset(self):
        self.text = ''
        self.prior_map = None
        self.langprob = None


    cdef set_verbose(self):
        self.verbose = True

    cdef set_alpha(self, float alpha):
        self.alpha = alpha

    cdef set_prior_map(self, prior_map):
        '''Set prior information about language probabilities.'''
        self.prior_map = [0.0] * len(self.langlist)
        cdef float sump = 0.0
        cdef float p = 0
        for i in xrange(len(self.prior_map)):
            lang = self.langlist[i]
            if lang in prior_map:
                p = prior_map[lang]
                if p < 0:
                    raise LangDetectException(ErrorCode.InitParamError, 'Prior probability must be non-negative.')
                self.prior_map[i] = p
                sump += p
        if sump <= 0.0:
            raise LangDetectException(ErrorCode.InitParamError, 'More one of prior probability must be non-zero.')
        for i in xrange(len(self.prior_map)):
            self.prior_map[i] /= sump

    cdef set_max_text_length(self, int max_text_length):
        '''Specify max size of target text to use for language detection.
        The default value is 10000(10KB).
        '''
        self.max_text_length = max_text_length

    def append(self, unicode text):
        '''Append the target text for language detection.
        If the total size of target text exceeds the limit size specified by
        Detector.set_max_text_length(int), the rest is cut down.
        '''
        text = self.URL_RE.sub(' ', text)
        text = self.MAIL_RE.sub(' ', text)
        text = NGram.normalize_vi(text)
        cdef unicode text_bytes = text
        cdef char space = <int>b' '[0]
        cdef string chars = self.text
        #chars += str(self.text)
        #chars.append(str(self.text))
        cdef int strlenx = min(len(text), self.max_text_length)
        print(strlenx)
        for uchar in text_bytes[:strlenx]:
            print(uchar)
            print(space)
            if (<int>&uchar) != space:# or (<int>pre) != space:
                #chars.append(ch)
                chars = chars + uchar
                #chars = chars + b'x'
            pre = uchar
        self.text = chars

    cdef cleaning_text(self):
        """Cleaning text to detect
        (eliminate URL, e-mail address and Latin sentence if it is not written in Latin alphabet).
        """
        cdef int latin_count = 0
        cdef int non_latin_count = 0
        cdef int i = 0
        cdef char ch
        cdef const char* text = self.text.c_str()
        cdef char badch = <char>b'\u0300'[0]
        for ch in text[:]:
            if 'A' <= ch <= 'z':
                latin_count += 1
            elif ch >= badch and unicode_block(ch) != 'Latin Extended Additional':
                non_latin_count += 1

        if latin_count * 2 < non_latin_count:
            text_without_latin = ''
            for ch in self.text:
                if ch < 'A' or 'z' < ch:
                    text_without_latin += ch
            self.text = text_without_latin

    def detect(self):
        '''Detect language of the target text and return the language name
        which has the highest probability.
        '''
        cdef list probabilities = self.get_probabilities()
        if probabilities:
            return probabilities[0].lang
        return self.UNKNOWN_LANG

    cdef list get_probabilities(self):
        if self.langprob is None:
            self._detect_block()
        return self._sort_probability(self.langprob)

    cdef _detect_block(self):
        self.cleaning_text()
        ngrams = self._extract_ngrams()
        if not ngrams:
            raise LangDetectException(ErrorCode.CantDetectError, 'No features in text.')

        self.langprob = numpy.zeros(len(self.langlist))#[0.0] * len(self.langlist)

        self.random.seed(self.seed)
        cdef float alpha
        cdef int i
        cdef np.ndarray prob
        for t in xrange(self.n_trial):
            prob = self._init_probability()
            alpha = self.alpha + self.random.gauss(0.0, 1.0) * self.ALPHA_WIDTH
            i = 0

            while True:
                self._update_lang_prob(prob, self.random.choice(ngrams), alpha)
                if i % 5 == 0:
                    prob = prob / sum(<double*>prob.data, prob.size)
                    if prob.max() > self.CONV_THRESHOLD or i >= self.ITERATION_LIMIT:
                        break
                    if self.verbose:
                        six.print_('>', self._sort_probability(prob))
                i += 1
            self.langprob += prob / self.n_trial
            # for j in xrange(len(self.langprob)):
            #     self.langprob[j] += prob[j] / self.n_trial
            if self.verbose:
                six.print_('==>', self._sort_probability(prob))

    cdef np.ndarray _init_probability(self):
        '''Initialize the map of language probabilities.
        If there is the specified prior map, use it as initial map.
        '''
        if self.prior_map is not None:
            return numpy.array(list(self.prior_map))
        else:
            return numpy.repeat(1.0 / len(self.langlist), len(self.langlist))#[1.0 / len(self.langlist)] * len(self.langlist)

    def _extract_ngrams(self):
        '''Extract n-grams from target text.'''
        RANGE = list(xrange(1, NGram.N_GRAM + 1))

        result = []
        ngram = NGram()
        cdef const char* c_str = self.text.c_str()
        print(self.text.length())
        cdef unsigned long ch
        for ch in c_str:
            ngram.add_char(ch)
            if ngram.capitalword:
                continue
            for n in RANGE:
                # optimized w = ngram.get(n)
                if len(ngram.grams) < n:
                    break
                w = ngram.grams[-n:]
                if w and w != ' ' and w in self.word_lang_prob_map:
                    result.append(w)
        return result

    def _update_lang_prob(self, prob, word, alpha):
        '''Update language probabilities with N-gram string(N=1,2,3).'''
        if word is None or word not in self.word_lang_prob_map:
            return False

        lang_prob_map = numpy.array(self.word_lang_prob_map[word])
        if self.verbose:
            six.print_('%s(%s): %s' % (word, self._unicode_encode(word), self._word_prob_to_string(lang_prob_map)))

        weight = alpha / self.BASE_FREQ
        prob *= weight + lang_prob_map
        # for i in xrange(len(prob)):
        #     prob[i] *= weight + lang_prob_map[i]
        return True

    def _word_prob_to_string(self, prob):
        result = ''
        for j in xrange(len(prob)):
            p = prob[j]
            if p >= 0.00001:
                result += ' %s:%.5f' % (self.langlist[j], p)
        return result

    cdef _normalize_prob(self, np.ndarray[np.float64_t, ndim=1] prob):
        '''Normalize probabilities and check convergence by the maximun probability.
        '''
        # maxp, sump = 0.0, np.sum(prob)
        prob = prob/sum(<double*>prob.data, prob.size)
        # for i in xrange(len(prob)):
        #     p = prob[i] / sump
        #     if maxp < p:
        #         maxp = p
        #     prob[i] = p
        return prob.max()

    cdef list _sort_probability(self, prob):
        result = [Language(lang, p) for (lang, p) in zip(self.langlist, prob) if p > self.PROB_THRESHOLD]
        result.sort(reverse=True)
        return result

    cdef char* _unicode_encode(self, word):
        buf = ''
        for ch in word:
            if ch >= six.u('\u0080'):
                st = hex(0x10000 + ord(ch))[2:]
                while len(st) < 4:
                    st = '0' + st
                buf += r'\u' + st[1:5]
            else:
                buf += ch
        return buf
