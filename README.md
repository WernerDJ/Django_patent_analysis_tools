# Patent Analysis Platform

This web page has been published as: [https://patent-analysis-e57ee02125bf.herokuapp.com/](https://patent-analysis-e57ee02125bf.herokuapp.com/)

This web platform provides data analysis tools tailored for intellectual property professionals. You can generate insights and visualizations from international patent data in just a few steps.

## How to Use

1.  Go to the [WIPO Patentscope](https://patentscope.wipo.int/search/en/search.jsf) search page.
2.  Sign in to your WIPO account to enable downloading results.
3.  Perform your patent search using any filters of interest (IPC codes, keywords, applicant, etc.).
4.  Download the result list as an Excel file (XLS format).
5.  Return here and upload the Excel file to generate custom visual analytics.

### Watch this short tutorial:

[![Tutorial Video](http://img.youtube.com/vi/0/0.jpg)](https://www.youtube.com/embed/eo31FWVMW_o?si=hmpVeIAeSrjxdBOC)

This website is still under development, so for now, there is no need to log in to access advanced features. Please be patient with the generation time of the figures, currently, this web uses only basic cloud resources. The web page might become unresponsive if you load more than 3000 rows of data, so please narrow down your results. This problem can be solved by usign more computing power, but that will depend on the interest arisen by this site. In future versions, signing up will allow users to save their searches and prevent the automatic deletion of generated graphics.

---

## Gallery of Sample Visualizations

Explore example graphics generated from international patent data.

### Countries Statistics

| Visualization                          | Description                            |
| :------------------------------------- | :------------------------------------- |
| ![Priority patent filling timeline](images/frequency_priority_years_r.png) | Priority patent filling, yearly frequency |
| ![Top Priority Countries](images/top_priority_countries_r.png) | Top Priority Filing Countries         |
| ![Top Countries](images/top_countries_r.png) | Top 10 countries by publication        |
| ![Origin-Destination Map](images/origin_destcountr_r.png) | Patent priority vs destination countries |

### Word frequency statistics: Wordclouds

| Visualization        | Description          |
| :------------------- | :------------------- |
| ![Noun Wordcloud](images/wcld_nouns_r.png) | Wordcloud (Nouns)    |
| ![Verb Wordcloud](images/wcld_verbs_r.png) | Wordcloud (Verbs)    |
| ![Adjective Wordcloud](images/wcld_adjectives_r.png) | Wordcloud (Adjectives) |

### IPC and Applicants Statistics

| Visualization                         | Description                                |
| :------------------------------------ | :----------------------------------------- |
| ![Top IPC Codes](images/top_ipcs_r.png) | Top 10 IPC groups                          |
| ![Parallel Coordinates](images/parallel_coordinates_r.png) | Technology evolution over time           |
| ![Top 20 Applicants](images/Top20Appl_r.png) | Top 20 Applicants                          |
| ![Top 5 Applicants timeline](images/Applicants_parallel_r.png) | Top 5 Applicants patent publication timeline |
| ![Most frequent IPC groups by Applicant](images/TopAppl_IPC_r.png) | Most frequent IPC groups by Applicant      |

### Technology Transfer Landscape

| Visualization                       | Description                                                        |
| :---------------------------------- | :----------------------------------------------------------------- |
| ![Inventor Transfer Network](images/network_plot.png) | Flow of Technological Know-How Through Shared Inventors<br>Applicants in the graph may have further connections out of this circle |
