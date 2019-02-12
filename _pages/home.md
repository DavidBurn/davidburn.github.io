---
title: "David Burn"
excerpt: "Data science, machine learning, functional programming and other projects"
layout: splash
permalink: /
header:
  overlay_image: /images/skyline.png
author_profile: true
feature_row:
  - image_path: /images/sheffield.png
    title: "Data Science"
    excerpt: "A collection of data science projects, research and competition entries"
    url: /data-science/
    btn_label: "Read More"
    btn_class: "btn--primary"
  - image_path: /images/splash/matrix.jpg
    title: "Code Challenges"
    excerprt: "Walkthroughs of my solutions to various coding challenges"
    url: /advent-2018/day1-2/
    btn_label: "Read More"
    btn_class: "btn--primary"
  - image_path: /images/skyline.jpg
    title: "Recent Posts"
    url: /index/
    btn_label: "Read More"
    btn_class: "btn--primary"
intro:
  - excerpt: "Some data science projects"
feature_row2:
  - image_path: /images/titanic.jpg
    title: "Titanic"
    excerpt: "Predicting survival with machine learning, a kaggle competition entry."
    url: "https://davidburn.github.io/titanic/"
    btn_label: "Read More"
    btn_class: "btn--primary"
feature_row3:
  - image_path: /images/wine.jpg
    title: "Wine Quality"
    excerpt: "Assessing the most important physiochemical attribute of wine when assigning a quality rating"
    url: "https://davidburn.github.io/wine-quality-feature-importance/"
    btn_label: "Read More"
    btn_class: "btn--primary"
---

{% include feature_row id="intro" type="center" %}

{% include feature_row %}

{% include feature_row id="feature_row2" type="left" %}

{% include feature_row id="feature_row3" type="right" %}
