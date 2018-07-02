FROM python:2.7-slim

RUN pip install dash==0.21.1 \
				dash-renderer==0.13.0 \
				dash-html-components==0.11.0 \
				dash-core-components==0.23.0 \
				plotly --upgrade \
				pandas \
				tables --upgrade \
				sklearn

RUN pip install scipy
RUN pip install datacleaner

EXPOSE 8050

RUN useradd -s /bin/bash nonroot
USER nonroot