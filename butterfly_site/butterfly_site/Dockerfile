FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY butterfly_site/ ./
RUN dotnet restore "ButterflySite.csproj"
RUN dotnet publish "ButterflySite.csproj" -c Release -o /app/publish

FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS final
WORKDIR /app
COPY --from=build /app/publish .
ENV ASPNETCORE_URLS=http://0.0.0.0:8080
EXPOSE 8080
ENTRYPOINT ["dotnet", "ButterflySite.dll"]
