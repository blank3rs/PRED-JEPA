import os
import time
import logging
import threading
import asyncio
import aiohttp
import psutil
import json
import hashlib
import sqlite3
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from queue import Queue
from PIL import Image
import io
import requests
from typing import List, Dict, Any
from datetime import datetime, timedelta

class DFSWebCrawler:
    def __init__(self, max_depth=10, max_threads=None, cache_dir='./crawler_cache'):
        self.max_depth = max_depth
        self.visited = set()
        self.data_lock = threading.Lock()
        self.url_stack = []
        self.thread_pool = None
        self.is_running = True
        self.loop = None
        self.cache_dir = cache_dir
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.join(cache_dir, 'images'), exist_ok=True)
        
        # Initialize SQLite cache
        self.init_db()
        
        # Auto-detect optimal thread count based on CPU and memory
        if max_threads is None:
            cpu_count = os.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024 ** 3)
            self.max_threads = min(int(cpu_count * 2), int(memory_gb * 2))
        else:
            self.max_threads = max_threads
            
        # Rate limiting per domain with adaptive delays
        self.domain_locks = {}
        self.domain_lock_lock = threading.Lock()
        self.domain_stats = {}
        
        # Metrics
        self.metrics = {
            'pages_crawled': 0,
            'bytes_downloaded': 0,
            'start_time': time.time(),
            'successful_requests': 0,
            'failed_requests': 0,
            'cached_hits': 0
        }
        
        # Content queues with size limits based on available memory
        memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        self.text_queue = Queue(maxsize=int(100000 * memory_gb))
        self.image_queue = Queue(maxsize=int(50000 * memory_gb))
        
        # Load visited URLs from cache
        self.load_visited_urls()
        
        logging.info(f"Initialized enhanced DFS crawler with {self.max_threads} threads and caching")

    def init_db(self):
        """Initialize SQLite database for caching"""
        self.db_path = os.path.join(self.cache_dir, 'crawler_cache.db')
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS pages
                        (url TEXT PRIMARY KEY, content TEXT, last_crawled TIMESTAMP)''')
            c.execute('''CREATE TABLE IF NOT EXISTS visited_urls
                        (url TEXT PRIMARY KEY, timestamp TIMESTAMP)''')
            conn.commit()

    def load_visited_urls(self):
        """Load previously visited URLs from cache"""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('SELECT url FROM visited_urls')
            self.visited = set(row[0] for row in c.fetchall())
            logging.info(f"Loaded {len(self.visited)} visited URLs from cache")

    def save_visited_url(self, url):
        """Save visited URL to cache"""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('INSERT OR REPLACE INTO visited_urls (url, timestamp) VALUES (?, ?)',
                     (url, datetime.now()))
            conn.commit()

    def get_cached_page(self, url):
        """Get page from cache if it exists and is not too old"""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('SELECT content, last_crawled FROM pages WHERE url = ?', (url,))
            result = c.fetchone()
            if result:
                content, last_crawled = result
                # Check if cache is less than 24 hours old
                if datetime.now() - datetime.fromisoformat(last_crawled) < timedelta(hours=24):
                    self.metrics['cached_hits'] += 1
                    return content
        return None

    def cache_page(self, url, content):
        """Cache page content"""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('INSERT OR REPLACE INTO pages (url, content, last_crawled) VALUES (?, ?, ?)',
                     (url, content, datetime.now()))
            conn.commit()

    def get_adaptive_delay(self, domain):
        """Get adaptive delay based on domain response times and errors"""
        stats = self.domain_stats.get(domain, {'errors': 0, 'success': 0})
        base_delay = 1.0
        
        # Increase delay if there are errors
        error_factor = 1 + (stats['errors'] * 0.5)
        
        # Decrease delay for successful requests
        success_factor = max(0.5, 1 - (stats['success'] * 0.1))
        
        return base_delay * error_factor * success_factor

    async def process_image(self, session, img_url):
        """Process and cache images"""
        try:
            # Generate cache path
            img_hash = hashlib.md5(img_url.encode()).hexdigest()
            cache_path = os.path.join(self.cache_dir, 'images', f'{img_hash}.jpg')
            
            # Check cache first
            if os.path.exists(cache_path):
                self.metrics['cached_hits'] += 1
                return Image.open(cache_path).convert('RGB')
            
            async with session.get(img_url, timeout=30) as response:
                if response.status == 200:
                    content = await response.read()
                    image = Image.open(io.BytesIO(content)).convert('RGB')
                    
                    # Cache the image
                    image.save(cache_path, 'JPEG', quality=85)
                    return image
                    
        except Exception as e:
            logging.error(f"Error processing image {img_url}: {e}")
            return None

    async def crawl_url_dfs(self, url, depth=0):
        if not self.is_running or depth > self.max_depth or url in self.visited:
            return

        domain = urlparse(url).netloc
        domain_lock = self.get_domain_lock(domain)

        with domain_lock:
            try:
                with self.data_lock:
                    if url in self.visited:
                        return
                    self.visited.add(url)
                    self.save_visited_url(url)
                    logging.info(f"Crawling URL: {url} at depth {depth}")

                # Check cache first
                cached_content = self.get_cached_page(url)
                if cached_content:
                    self.process_cached_content(url, cached_content, depth)
                    return

                # Adaptive delay based on domain stats
                delay = self.get_adaptive_delay(domain)
                await asyncio.sleep(delay)

                async with aiohttp.ClientSession() as session:
                    try:
                        async with session.get(url, headers={
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                            'Accept-Language': 'en-US,en;q=0.5',
                            'Connection': 'keep-alive',
                            'Upgrade-Insecure-Requests': '1'
                        }, timeout=30) as response:
                            if response.status != 200:
                                self.update_domain_stats(domain, success=False)
                                logging.warning(f"Got status {response.status} for URL: {url}")
                                return

                            content_type = response.headers.get('content-type', '')
                            content = await response.read()
                            
                            self.metrics['bytes_downloaded'] += len(content)
                            self.metrics['pages_crawled'] += 1
                            self.update_domain_stats(domain, success=True)

                            # Cache the content
                            self.cache_page(url, content.decode('utf-8', errors='ignore'))

                            if 'text/html' in content_type:
                                soup = BeautifulSoup(content, 'html.parser')
                                
                                # Extract and process links first (DFS)
                                links = []
                                for link in soup.find_all('a'):
                                    href = link.get('href')
                                    if href:
                                        full_url = urljoin(url, href)
                                        if self.is_valid_url(full_url) and self.should_crawl_url(full_url):
                                            links.append(full_url)
                                
                                # Process current page content
                                text = self.extract_text(soup)
                                if text and len(text.split()) > 50:  # Only keep substantial text
                                    if not self.text_queue.full():
                                        self.text_queue.put({'url': url, 'text': text, 'depth': depth})
                                        logging.info(f"Added text from {url} to queue. Queue size: {self.text_queue.qsize()}")
                                
                                # Extract and process images
                                for img in soup.find_all('img'):
                                    if not self.is_running:
                                        return
                                        
                                    img_url = img.get('src')
                                    if img_url and self.is_valid_image_url(img_url):
                                        img_url = urljoin(url, img_url)
                                        processed_image = await self.process_image(session, img_url)
                                        if processed_image and not self.image_queue.full():
                                            self.image_queue.put({
                                                'url': img_url,
                                                'image': processed_image,
                                                'depth': depth
                                            })
                                            logging.info(f"Added image from {img_url} to queue. Queue size: {self.image_queue.qsize()}")
                                
                                # Process links with adaptive delays
                                for link in links:
                                    if not self.is_running:
                                        return
                                    await asyncio.sleep(self.get_adaptive_delay(urlparse(link).netloc))
                                    await self.crawl_url_dfs(link, depth + 1)

                    except aiohttp.ClientError as e:
                        self.update_domain_stats(domain, success=False)
                        logging.error(f"Network error crawling {url}: {e}")
                        await asyncio.sleep(5)  # Back off on network error

            except Exception as e:
                logging.error(f"Error crawling {url}: {e}")
                await asyncio.sleep(5)  # Back off on error

    def update_domain_stats(self, domain, success=True):
        """Update domain statistics for adaptive rate limiting"""
        if domain not in self.domain_stats:
            self.domain_stats[domain] = {'errors': 0, 'success': 0}
        
        if success:
            self.domain_stats[domain]['success'] += 1
            self.metrics['successful_requests'] += 1
        else:
            self.domain_stats[domain]['errors'] += 1
            self.metrics['failed_requests'] += 1

    def should_crawl_url(self, url):
        """Additional URL filtering"""
        parsed = urlparse(url)
        # Skip social media, ads, tracking, etc.
        skip_domains = {'facebook.com', 'twitter.com', 'instagram.com', 'ads.', 'analytics.', 'tracker.'}
        return not any(domain in parsed.netloc for domain in skip_domains)

    def process_cached_content(self, url, content, depth):
        """Process content from cache"""
        try:
            soup = BeautifulSoup(content, 'html.parser')
            text = self.extract_text(soup)
            if text and len(text.split()) > 50:
                if not self.text_queue.full():
                    self.text_queue.put({'url': url, 'text': text, 'depth': depth})
                    logging.info(f"Added cached text from {url} to queue")
        except Exception as e:
            logging.error(f"Error processing cached content from {url}: {e}")

    def get_enhanced_metrics(self) -> Dict[str, Any]:
        """Get enhanced metrics including cache and error stats"""
        metrics = self.get_metrics()
        metrics.update({
            'cache_hits': self.metrics['cached_hits'],
            'successful_requests': self.metrics['successful_requests'],
            'failed_requests': self.metrics['failed_requests'],
            'success_rate': (self.metrics['successful_requests'] / 
                           (self.metrics['successful_requests'] + self.metrics['failed_requests'])
                           if self.metrics['successful_requests'] + self.metrics['failed_requests'] > 0 else 0),
            'domain_stats': self.domain_stats
        })
        return metrics

    def get_domain_lock(self, domain):
        with self.domain_lock_lock:
            if domain not in self.domain_locks:
                self.domain_locks[domain] = threading.Lock()
            return self.domain_locks[domain]

    def run_async_loop(self, seed_urls):
        """Run the async event loop in the current thread"""
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            # Create tasks for each seed URL
            tasks = []
            for url in seed_urls:
                logging.info(f"Starting crawl for seed URL: {url}")
                tasks.append(self.crawl_url_dfs(url))
            
            # Run tasks
            self.loop.run_until_complete(asyncio.gather(*tasks))
            
        except Exception as e:
            logging.error(f"Error in crawler async loop: {e}")
        finally:
            self.loop.close()

    def start_crawling(self, seed_urls: List[str]):
        """Start the crawler in a separate thread"""
        logging.info(f"Starting crawler with {len(seed_urls)} seed URLs")
        self.thread_pool = threading.Thread(target=self.run_async_loop, args=(seed_urls,))
        self.thread_pool.start()

    @staticmethod
    def is_valid_url(url: str) -> bool:
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

    @staticmethod
    def is_valid_image_url(url: str) -> bool:
        return any(url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp'])

    @staticmethod
    def is_valid_video_url(url: str) -> bool:
        video_platforms = ['youtube.com', 'vimeo.com', 'dailymotion.com']
        return any(platform in url.lower() for platform in video_platforms)

    @staticmethod
    def extract_text(soup: BeautifulSoup) -> str:
        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()
        
        # Get text from paragraphs
        paragraphs = soup.find_all('p')
        text = ' '.join(p.get_text().strip() for p in paragraphs)
        return text

    def get_metrics(self) -> Dict[str, Any]:
        elapsed_time = time.time() - self.metrics['start_time']
        return {
            'pages_crawled': self.metrics['pages_crawled'],
            'bytes_downloaded': self.metrics['bytes_downloaded'],
            'elapsed_time': elapsed_time,
            'crawl_rate': self.metrics['pages_crawled'] / elapsed_time if elapsed_time > 0 else 0,
            'memory_usage': psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        }

    def stop(self):
        """Stop the crawler gracefully"""
        logging.info("Stopping crawler...")
        self.is_running = False
        if self.loop:
            self.loop.stop()
        if self.thread_pool and self.thread_pool.is_alive():
            self.thread_pool.join()
        logging.info("Crawler stopped") 