import pandas as pd

d = {'url':'https://www.numbeo.com/quality-of-life/rankings_by_country.jsp',
     'name': 'rankings_by_country',
     'load_func': lambda URL: pd.read_html(URL)[1]}

df = d['load_func'](d['url'])

continent_mapping = {
    'Europe': ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Estonia', 'Finland',
               'France', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy', 'Latvia', 'Lithuania',
               'Netherlands', 'Norway', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain',
               'Sweden', 'United Kingdom', 'Luxembourg', 'Denmark', 'Switzerland'],
    'Africa': ['Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cameroon',
               'Central African Republic', 'Chad', 'Comoros', 'Congo', 'Djibouti', 'Egypt', 'Equatorial Guinea',
               'Eritrea', 'Eswatini', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau',
               'Ivory Coast', 'Kenya', 'Lesotho', 'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali',
               'Mauritania', 'Mauritius', 'Morocco', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Rwanda',
               'Sao Tome and Principe', 'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia', 'South Africa',
               'South Sudan', 'Sudan', 'Tanzania', 'Togo', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe'],
    'Asia': ['Afghanistan', 'Armenia', 'Azerbaijan', 'Bahrain', 'Bangladesh', 'Bhutan', 'Brunei', 'Cambodia',
             'China', 'Georgia', 'India', 'Indonesia', 'Iran', 'Iraq', 'Israel', 'Japan', 'Jordan', 'Kazakhstan',
             'Kuwait', 'Kyrgyzstan', 'Laos', 'Lebanon', 'Malaysia', 'Maldives', 'Mongolia', 'Myanmar', 'Nepal',
             'North Korea', 'Oman', 'Pakistan', 'Palestine', 'Philippines', 'Qatar', 'Saudi Arabia', 'Singapore',
             'South Korea', 'Sri Lanka', 'Syria', 'Taiwan', 'Tajikistan', 'Thailand', 'Timor-Leste', 'Turkey',
             'Turkmenistan', 'United Arab Emirates', 'Uzbekistan', 'Vietnam', 'Yemen'],
    'Oceania': ['Australia', 'Fiji', 'Kiribati', 'Marshall Islands', 'Micronesia', 'Nauru', 'New Zealand', 'Palau',
                'Papua New Guinea', 'Samoa', 'Solomon Islands', 'Tonga', 'Tuvalu', 'Vanuatu'],
    'North America': ['Antigua and Barbuda', 'Bahamas', 'Barbados', 'Belize', 'Canada', 'Costa Rica', 'Cuba',
                      'Dominica', 'Dominican Republic', 'El Salvador', 'Grenada', 'Guatemala', 'Haiti', 'Honduras',
                      'Jamaica', 'Mexico', 'Nicaragua', 'Panama', 'Saint Kitts and Nevis', 'Saint Lucia',
                      'Saint Vincent and the Grenadines', 'Trinidad and Tobago', 'United States'],
    'South America': ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Guyana', 'Paraguay',
                      'Peru', 'Suriname', 'Uruguay', 'Venezuela']
}

continent_df = pd.DataFrame([
    {'Country': country, 'Continent': continent}
    for continent, countries in continent_mapping.items()
    for country in countries
])

df = df.merge(continent_df, on='Country', how='left')

### find top 5 countries by Quality of Life Index, excluding Luxembourg, Denmark, Switzerland

excluded_countries = ['Luxembourg', 'Denmark', 'Switzerland']
top_5_quality_of_life = df[~df['Country'].isin(excluded_countries)].nlargest(5, 'Quality of Life Index')
print(top_5_quality_of_life)

### calculate average Quality of Life Index in Europe*

avg_quality_of_life_europe = df[df['Continent'] == 'Europe']['Quality of Life Index'].mean()
print(avg_quality_of_life_europe)

### calculate average Quality of Life Index in Africa*

avg_quality_of_life_africa = df[df['Continent'] == 'Africa']['Quality of Life Index'].mean()
print(avg_quality_of_life_africa)

q1 = df['Quality of Life Index'].quantile(0.33)
q2 = df['Quality of Life Index'].quantile(0.67)

### try to split countries into segments using approach you prefer

df['Segment'] = pd.cut(df['Quality of Life Index'],
                       bins=[-float('inf'), q1, q2, float('inf')],
                       labels=['Low', 'Medium', 'High'])
segments = df.groupby('Segment')['Country'].apply(list)
print(segments)


### * there's no "Continent" column, try to join on your own (см.выше)
