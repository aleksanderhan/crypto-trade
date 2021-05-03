
-- Relational tables

CREATE TABLE biz_op (
	id INTEGER PRIMARY KEY,
	ts TIMESTAMPTZ NOT NULL,
);

CREATE TABLE biz_reply (
	id INTEGER PRIMARY KEY,
	op_id INTEGER NOT NULL REFERENCES biz_op(id),
	ts TIMESTAMPTZ NOT NULL
);

CREATE TABLE keyword (
	id SERIAL PRIMARY KEY,
	keyword VARCHAR NOT NULL
);

CREATE TABLE biz_op_keyword (
	id SERIAL PRIMARY KEY,
	op_id INTEGER REFERENCES biz_op(id),
	keyword_id INTEGER REFERENCES keyword(id),
	freq INTEGER NOT NULL
);

CREATE TABLE biz_reply_keyword (
	id SERIAL PRIMARY KEY,
	reply_id INTEGER REFERENCES biz_reply(id),
	keyword_id INTEGER REFERENCES keyword(id),
	freq INTEGER NOT NULL
);



-- Timeseries tables

CREATE TABLE biz_sia_sentiment (
	ts TIMESTAMPTZ NOT NULL,
	positive DECIMAL NOT NULL,
	negative DECIMAL NOT NULL,
	neutral DECIMAL NOT NULL,
	compound DECIMAL NOT NULL,
	type VARCHAR NOT NULL,
	post_id INTEGER NOT NULL
);
SELECT create_hypertable('biz_sia_sentiment', 'ts');