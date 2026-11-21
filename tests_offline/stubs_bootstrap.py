"""离线 stub：在无第三方依赖的环境下模拟 numpy/redis/sklearn/sqlalchemy/fastapi。"""

# ---------- numpy 最小替身 ----------
def array(values, dtype=None):
    return values

class _Numpy:
    array = staticmethod(array)

# ---------- redis 最小替身 ----------
from collections import defaultdict

class Redis:
    _instances = {}

    @classmethod
    def from_url(cls, url, decode_responses=False):
        return cls._instances.setdefault(url, cls())

    def __init__(self):
        self.queues = defaultdict(list)

    def rpush(self, key, value):
        self.queues[key].append(value)

    def blpop(self, key, timeout=0):
        items = self.queues[key]
        return (key, items.pop(0)) if items else None

# ---------- sklearn 最小替身 ----------
class StandardScaler:
    def fit_transform(self, x):
        return x

class LogisticRegression:
    def __init__(self, max_iter=500, random_state=None, multi_class=None):
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, x, y):
        self.classes_ = sorted(set(y))
        self.width_ = len(x[0])
        return self

    def _bucket(self, value):
        # 与训练标签同序的四档阈值
        if value < (12 if self.width_ == 14 else 5):
            return "normal"
        if value < (24 if self.width_ == 14 else 7):
            return "watch"
        if value < (40 if self.width_ == 14 else 9):
            return "major"
        return "critical"

    def predict(self, x):
        row = x[0]
        # 现行模型（14 维，含交叉特征）按平均停留判定；候选模型（12 维）按箱量判定
        value = row[1] if self.width_ == 14 else row[0]
        return [self._bucket(value)]

    def predict_proba(self, x):
        n = len(self.classes_)
        pred = self.predict(x)[0]
        probs = [0.7 if c == pred else 0.3 / (n - 1) for c in self.classes_]
        total = sum(probs)
        return [[p / total for p in probs]]

class Pipeline:
    def __init__(self, steps):
        self._steps = dict(steps)

    def fit(self, x, y):
        for step in self._steps.values():
            if hasattr(step, "fit"):
                step.fit(x, y)
        return self

    @property
    def named_steps(self):
        return self._steps

    def predict(self, x):
        return self._steps["lr"].predict(x)

    def predict_proba(self, x):
        return self._steps["lr"].predict_proba(x)

# ---------- SQLAlchemy 最小替身：内存表 + 条件查询 ----------
import copy as _copy

class _Col:
    def __init__(self, default=None, pk=False, nullable=True):
        self.default = default
        self.pk = pk
        self.name = None
        self.model = None

    def __set_name__(self, owner, name):
        self.name = name
        self.model = owner

    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        if self.name not in obj.__dict__:
            default = self.default
            obj.__dict__[self.name] = default() if callable(default) else default
        return obj.__dict__[self.name]

    def __set__(self, obj, value):
        obj.__dict__[self.name] = value

    def __eq__(self, other):
        if isinstance(other, _Col):
            return ("col_eq", self.name, other.name)
        return ("eq", self.name, other)

    def __ne__(self, other):
        return ("ne", self.name, other)

    def in_(self, values):
        return ("in", self.name, list(values))

    def is_not(self, value):
        return ("is_not", self.name, value)

    def desc(self):
        return ("desc", self.name)

    def asc(self):
        return ("asc", self.name)

def mapped_column(*args, **kwargs):
    return _Col(default=kwargs.get("default"), pk=kwargs.get("primary_key", False),
                nullable=kwargs.get("nullable", True))

def relationship(*args, **kwargs):
    return _Col(default=None)

class _Type:
    def __init__(self, *a, **k):
        pass

class _TypeMeta(type):
    def __call__(cls, *a, **k):
        return cls

class Boolean(metaclass=_TypeMeta): pass
class DateTime(metaclass=_TypeMeta): pass
class Float(metaclass=_TypeMeta): pass
class Integer(metaclass=_TypeMeta): pass
class JSON(metaclass=_TypeMeta): pass
class String(metaclass=_TypeMeta): pass
class Text(metaclass=_TypeMeta): pass

def ForeignKey(target):
    return ("fk", target)

class UniqueConstraint:
    def __init__(self, *a, **k):
        self.cols = a

class _Select:
    def __init__(self, entities):
        self.entities = entities
        self.filters = []
        self.joins = []
        self.orders = []

    def where(self, *filters):
        self.filters.extend(filters)
        return self

    def join(self, entity, onclause):
        self.joins.append((entity, onclause))
        return self

    def order_by(self, *cols):
        self.orders.extend(cols)
        return self

def select(*entities):
    return _Select(entities)

def func():
    raise NotImplementedError

class _Result:
    def __init__(self, rows, multi):
        self._rows = rows
        self._multi = multi
        self._scalar_mode = False

    def scalars(self):
        self._scalar_mode = True
        return self

    def all(self):
        if self._multi and self._scalar_mode:
            return [r[0] for r in self._rows]
        return list(self._rows)

    def first(self):
        if not self._rows:
            return None
        return self._rows[0][0] if self._multi else self._rows[0]

    def scalar_one_or_none(self):
        return self.first()

    def scalar_one(self):
        if not self._rows:
            raise LookupError("no rows")
        return self.first()

class Engine:
    def __init__(self, url):
        self.url = url
        self.tables = {}
        self.next_ids = {}

    def table(self, cls):
        return self.tables.setdefault(cls, {})

    def next_id(self, cls):
        value = self.next_ids.get(cls, 0) + 1
        self.next_ids[cls] = value
        return value

class Session:
    def __init__(self, engine):
        self.engine = engine
        self.pending = []
        self.snapshots = {}

    def add(self, obj):
        self.pending.append(obj)

    def flush(self):
        for obj in self.pending:
            cls = type(obj)
            if getattr(obj, "id", None) is None:
                obj.id = self.engine.next_id(cls)
            self.engine.table(cls)[obj.id] = obj
        self.pending.clear()

    def commit(self):
        self.flush()
        for cls, rows in self.engine.tables.items():
            for obj_id, obj in rows.items():
                self.snapshots[(cls, obj_id)] = _copy.deepcopy(obj.__dict__)

    def rollback(self):
        for (cls, obj_id), snapshot in self.snapshots.items():
            obj = self.engine.tables.get(cls, {}).get(obj_id)
            if obj is not None:
                obj.__dict__ = _copy.deepcopy(snapshot)
        self.pending.clear()

    def close(self):
        pass

    def get(self, cls, obj_id):
        return self.engine.table(cls).get(obj_id)

    def scalar(self, stmt):
        result = self._execute(stmt)
        return result.first()

    def execute(self, stmt):
        return self._execute(stmt)

    def _match(self, obj, token):
        kind, name, value = token
        attr = getattr(obj, name)
        if kind == "eq":
            return attr == value
        if kind == "ne":
            return attr != value
        if kind == "in":
            return attr in value
        if kind == "is_not":
            return attr is not value
        raise AssertionError(f"unsupported filter {kind}")

    def _execute(self, stmt):
        primary = stmt.entities[0]
        col_entity = isinstance(primary, _Col)
        entity_cls = primary.model if col_entity else primary
        rows = list(self.engine.table(entity_cls).values())
        rows = [r for r in rows if all(self._match(r, f) for f in stmt.filters)]
        for entity, onclause in stmt.joins:
            kind, left_name, right_name = onclause
            assert kind == "col_eq"
            joined = []
            # 若 onclause 左侧属于被 join 的实体（如 join(DelayCase, DelayCase.id==TrialResult.case_id)）
            left_is_joined = any(getattr(j, left_name, None) is not None
                                 for j in list(self.engine.table(entity).values())[:1])
            for row in rows:
                if left_is_joined:
                    match = next((j for j in self.engine.table(entity).values()
                                  if getattr(j, left_name) == getattr(row, right_name)), None)
                else:
                    match = next((j for j in self.engine.table(entity).values()
                                  if getattr(j, right_name) == getattr(row, left_name)), None)
                if match is not None:
                    joined.append((row, match))
            rows = joined
        if stmt.orders:
            def sort_key(row):
                target = row[0] if isinstance(row, tuple) else row
                return getattr(target, name)
            for order in reversed(stmt.orders):
                kind, name = order if isinstance(order, tuple) else ("asc", order.name)
                rows.sort(key=sort_key, reverse=(kind == "desc"))
        if col_entity:
            rows = [(getattr(r, primary.name),) for r in rows]
        elif len(stmt.entities) > 1:
            rows = [r if isinstance(r, tuple) else (r,) for r in rows]
        elif rows and not isinstance(rows[0], tuple):
            rows = [(r,) for r in rows]
        return _Result(rows, multi=True)

def create_engine(url, **kwargs):
    return Engine(url)

def sessionmaker(bind=None, **kwargs):
    engine = bind
    return lambda: Session(engine)

class DeclarativeBase:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        def __init__(self, **values):
            for key, value in values.items():
                setattr(self, key, value)

        cls.__init__ = __init__

# ---------- fastapi 最小替身 ----------
class HTTPException(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
