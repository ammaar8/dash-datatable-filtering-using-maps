Demo for filtering Dash Datatable using Mapbox Scatter map
---
![https://youtu.be/WASIJDCHdcM](thumbnail.png)
I was trying to use a mapbox scatter plot to interactively filter data in a dash datatable and was having some problem with it because I was actually updating the data by removing the points which were not selected. This caused the issue where if filtered rows were selected and the maps selection was changed, the selections would become messed up. I later realized this may be because the number of rows was changing and the data table could not keep track of the selected rows so came up with a work around -

To make sure the number of rows remain the same and take advantage of native filtering and selections, I simply added an extra column to the data which has true/false based on whether the point was in the selection or not. This allowed me to make use of native filtering by using `contains true` to filter based on the map. Now the thing left to do was to automate this so the user doesn't have to input true to get filtering. To do this I simply put `{OnMap} contains true` using a callback which ensured the user doesn't have to fill that in. Finally I used the styling to keep the column always hidden to the user and that's it! Filter rows using Map Selections. Hope this is useful to people if they get stuck on this!

There are comments in the code for a more detailed explanation!