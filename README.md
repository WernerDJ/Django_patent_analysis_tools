# Patent Analysis Platform

The web page has been published at [https://patent-analysis-e57ee02125bf.herokuapp.com/](https://patent-analysis-e57ee02125bf.herokuapp.com/).

<div style="display: flex; align-items: flex-start; margin-bottom: 1.5em;">
  <img src="static/images/dfp.jpg" alt="Patent Analysis Graphic" style="width: 50%; height: auto; margin-right: 1.5em; border: 1px solid #ccc; border-radius: 6px;">
  <div>
    <h1 style="font-size: 2.2em; margin-bottom: 0.5em;">Welcome to the Patent Analysis Platform</h1>
    <p style="font-size: 1.2em; line-height: 1.6;">
      This web platform provides data analysis tools tailored for intellectual property professionals.
      You can generate insights and visualizations from international patent data in just a few steps.
    </p>
  </div>
</div>

## How to Use

1. Go to the [WIPO Patentscope](https://patentscope.wipo.int/search/en/search.jsf) search page.
2. Sign in to your WIPO account to enable downloading results.
3. Perform your patent search using any filters of interest (IPC codes, keywords, applicant, etc.).
4. Download the result list as an Excel file (XLS format).
5. Return here and upload the Excel file to generate custom visual analytics.

<div style="margin-top: 2em;">
  <h3 style="font-size: 1.6em;">Watch this short tutorial:</h3>
  <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; margin-top: 1em; border: 1px solid #ccc; border-radius: 8px;">
    <iframe src="https://www.youtube.com/embed/eo31FWVMW_o?si=hmpVeIAeSrjxdBOC" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
  </div>
</div>

<p style="font-size: 1.2em; line-height: 1.6;">
  This website is still under development, so for now, there is no need to log in to access advanced features.
  Please be patient with the generation time of the figures, currently, this web uses only basic cloud resources.
  The web page might become unresponsive if you load more than 3000 rows of data, so please narrow down your results.
  This problem can be solved by using more computing power, but that will depend on the interest arisen by this site.
  In future versions, signing up will allow users to save their searches and prevent the automatic deletion of generated graphics.
</p>

<hr style="margin: 3em 0;">

## Gallery of Sample Visualizations

Explore example graphics generated from international patent data.

### Countries Statistics

<div style="display: flex; flex-wrap: wrap; gap: 1em;">
  <div style="flex: 1 1 48%;">
    <img src="static/images/frequency_priority_years_r.png" alt="priority patent filling timeline" style="width: 60%; height: auto; border: 1px solid #ccc; border-radius: 6px;">
    <p style="text-align: center;">Priority patent filling, yearly frequency</p>
  </div>
  <div style="flex: 1 1 48%;">
    <img src="static/images/top_priority_countries_r.png" alt="Top Priority Countries" style="width: 60%; height: auto; border: 1px solid #ccc; border-radius: 6px;">
    <p style="text-align: center;">Top Priority Filing Countries</p>
  </div>
  <div style="flex: 1 1 48%;">
    <img src="static/images/top_countries_r.png" alt="Top Countries" style="width: 60%; height: auto; border: 1px solid #ccc; border-radius: 6px;">
    <p style="text-align: center;">Top 10 countries by publication</p>
  </div>
  <div style="flex: 1 1 48%;">
    <img src="static/images/origin_destcountr_r.png" alt="Origin-Destination Map" style="width: 60%; height: auto;  border: 1px solid #ccc; border-radius: 6px;">
    <p style="text-align: center;">Patent priority vs destination countries</p>
  </div>
</div>

### Word Frequency Statistics: Wordclouds

<div style="display: flex; flex-wrap: wrap; gap: 1em;">
  <div style="flex: 1 1 48%;">
    <img src="static/images/wcld_nouns_r.png" alt="Noun Wordcloud" style="width: 60%;  height: auto; border: 1px solid #ccc; border-radius: 6px;">
    <p style="text-align: center;">Wordcloud (Nouns)</p>
  </div>
  <div style="flex: 1 1 48%;">
    <img src="static/images/wcld_verbs_r.png" alt="Verb Wordcloud" style="width: 60%; height: auto;  border: 1px solid #ccc; border-radius: 6px;">
    <p style="text-align: center;">Wordcloud (Verbs)</p>
  </div>
  <div style="flex: 1 1 48%;">
    <img src="static/images/wcld_adjectives_r.png" alt="Adjective Wordcloud" style="width: 60%; height: auto;  border: 1px solid #ccc; border-radius: 6px;">
    <p style="text-align: center;">Wordcloud (Adjectives)</p>
  </div>
</div>

### IPC and Applicants Statistics

<div style="display: flex; flex-wrap: wrap; gap: 1em;">
  <div style="flex: 1 1 48%;">
    <img src="static/images/top_ipcs_r.png" alt="Top IPC Codes" style="width: 60%; height: auto;  border: 1px solid #ccc; border-radius: 6px;">
    <p style="text-align: center;">Top 10 IPC groups</p>
  </div>
  <div style="flex: 1 1 48%;">
    <img src="static/images/parallel_coordinates_r.png" alt="Parallel Coordinates" style="width: 60%; height: auto;  border: 1px solid #ccc; border-radius: 6px;">
    <p style="text-align: center;">Technology evolution over time</p>
  </div>
  <div style="display: flex; flex-wrap: wrap; gap: 1em;">
    <div style="flex: 1 1 48%;">
      <img src="static/images/Top20Appl_r.png" alt="Top 20 Applicants" style="width: 60%; height: auto;  border: 1px solid #ccc; border-radius: 6px;">
      <p style="text-align: center;">Top 20 Applicants</p>
    </div>
    <div style="flex: 1 1 48%;">
      <div style="display: flex; flex-wrap: wrap; gap: 1em;">
        <div style="flex: 1 1 48%;">
          <img src="static/images/Applicants_parallel_r.png" alt="Top 5 Applicants timeline" style="width: 60%; height: auto;  border: 1px solid #ccc; border-radius: 6px;">
          <p style="text-align: center;">Top 5 Applicants patent publication timeline</p>
        </div>
        <img src="static/images/TopAppl_IPC_r.png" alt="Most frequent IPC groups by Applicant" style="width: 60%; height: auto;  border: 1px solid #ccc; border-radius: 6px;">
        <p style="text-align: center;">Most frequent IPC groups by Applicant</p>
      </div>
    </div>
  </div>
</div>

### Technology Transfer Landscape

<div style="display: flex; flex-wrap: wrap; gap: 1em;">
  <div style="flex: 1 1 48%;">
    <img src="static/images/network_plot.png" alt="Inventor Transfer Network" style="width: 100%; border: 1px solid #ccc; border-radius: 6px;">
    <p style="text-align: center;">Flow of Technological Know-How Through Shared Inventors</p>
    <p style="text-align: center;">Applicants in the graph may have further connections out of this circle</p>
  </div>
</div>
 |

### Technology Transfer Landscape

| Visualization                       | Description                                                        |
| :---------------------------------- | :----------------------------------------------------------------- |
| ![Inventor Transfer Network](static/images/network_plot.png) | Flow of Technological Know-How Through Shared Inventors<br>Applicants in the graph may have further connections out of this circle |
